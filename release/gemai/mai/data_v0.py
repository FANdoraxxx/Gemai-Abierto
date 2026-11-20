"""LEGACY reference copy of the pre-optimization loader (used only by tools/test_data_equiv.py and tools/prof_loader.py --impl v0).

Windowed training data built from work/charts.pkl (+ work/meta.json, work/mel/*.npy).  See docs/MODEL_SPEC.md.

Sequence layout:
  BOS LV SLOT TYPE TOUCH EXN FEST DES VER BPM PROG CTX <history events> TGT <target events> EOS
cur_time[i] = tick (relative to the window start s) of the latest time token at/before position i
(-768 before the first history time token, 0 at TGT).  loss_mask[i] = 1 where tokens[i+1] is a target token / EOS.
"""
from __future__ import annotations
import os, json, pickle, math
from collections import OrderedDict
import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset, Dataset, get_worker_info
except ImportError:                      # allows light unit tests in the CPU-only VM (no torch)
    torch = None

    class IterableDataset:               # type: ignore
        pass

    class Dataset:                       # type: ignore
        pass

    def get_worker_info():
        return None

from .tokens import Vocab, TPB, T_MIN, T_MAX, L_BEATS
from .tempo import NpTempo
from .audio import beat_sync, FPS

BAR = 4 * TPB                    # 192 ticks
HIST = -T_MIN                    # 768
TGT = T_MAX                      # 768
LOOK = L_BEATS * TPB             # 192
N_FRAMES = HIST + TGT + LOOK     # 1728 audio frames (ticks s-768 .. s+959)
N_MEL = 96
MAX_LEN = 1024
HEADER = 12                      # BOS + 8 cond + BPM + PROG + CTX
MODES = ("N", "H", "V", "HV")      # all modes present in charts.pkl
TRAIN_MODES = ("N", "H")           # only left-right mirroring is used for training (V/HV break playing habits)
COND_P = 0.3
INVALID = -4.0


def designer_names(meta):
    d = meta["designers"]
    if isinstance(d, dict):
        items = sorted(d.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)
        return [k for k, _ in items]
    out = []
    for x in d:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            out.append(x.get("name", x.get("des", "")))
        else:
            out.append(x[0])
    return out


def make_vocab(meta):
    return Vocab(designer_names(meta))


def _rnd(x):
    return int(math.floor(x + 0.5))


# ----------------------------------------------------------------------------------------------
class _Source:
    """Lazily loaded chart store.  The big pickle is loaded inside each worker process (never pickled across)."""
    _LAZY = ("charts", "V", "nbars", "cum", "_tempo", "_mels", "_cond", "_unk", "_arr_cache")

    def __init__(self, data, meta, mel_dir, split, max_len=MAX_LEN, cond_p=COND_P):
        self.data, self.meta, self.mel_dir = str(data), str(meta), str(mel_dir)
        self.split, self.max_len, self.cond_p = split, max_len, cond_p
        self._loaded = False

    def __getstate__(self):
        d = self.__dict__.copy()
        for k in self._LAZY:
            d.pop(k, None)
        d["_loaded"] = False
        return d

    # ---------------------------------------------------------------- loading
    def _load(self):
        if self._loaded:
            return
        with open(self.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.V = make_vocab(meta)
        with open(self.data, "rb") as f:
            allc = pickle.load(f)
        self.charts = [c for c in allc if c.get("split") == self.split]
        del allc
        self.nbars = np.array([int(c["last_tick"]) // BAR + 1 for c in self.charts], dtype=np.float64)
        self.cum = np.cumsum(self.nbars)
        self.cum /= self.cum[-1]
        self._tempo, self._cond = {}, {}
        self._mels = OrderedDict()
        V = self.V
        self._unk = np.array([V["LV_UNK"], V["SLOT_UNK"], V["TYPE_UNK"], V["TOUCH_UNK"], V["EXN_UNK"],
                              V["FEST_UNK"], V["DES_UNK"], V["VER_UNK"]], dtype=np.int32)
        self._loaded = True

    # ---------------------------------------------------------------- per-chart helpers
    @staticmethod
    def _arrays(c, mode):
        if "modes" in c:
            d = c["modes"][mode]
            return d["ticks"], d["offs"], d["toks"]
        if mode in c and isinstance(c[mode], dict):
            d = c[mode]
            return d["ticks"], d["offs"], d["toks"]
        if "ticks_" + mode in c:
            return c["ticks_" + mode], c["offs_" + mode], c["toks_" + mode]
        if isinstance(c.get("ticks"), dict):
            return c["ticks"][mode], c["offs"][mode], c["toks"][mode]
        raise KeyError("chart has no arrays for mode " + mode)

    def _tm(self, ci):
        tm = self._tempo.get(ci)
        if tm is None:
            c = self.charts[ci]
            tm = self._tempo[ci] = NpTempo(c["beats"], c["bpms"])
        return tm

    def _cond_tokens(self, ci):
        t = self._cond.get(ci)
        if t is not None:
            return t
        c, V = self.charts[ci], self.V
        lv = c.get("lv")
        if lv is None or (isinstance(lv, float) and math.isnan(lv)):
            lvt = V["LV_UNK"]
        else:
            lvt = V[f"LV_{min(max(_rnd(lv * 10) - 10, 0), 140)}"]
        slot = int(c["slot"]) - 2
        slt = V[f"SLOT_{slot}"] if 0 <= slot < 5 else V["SLOT_UNK"]
        typ = V["TYPE_DX"] if c.get("dx") else V["TYPE_STD"]
        yn = lambda name, f: V[f"{name}_{'Y' if f else 'N'}"]
        des = int(c.get("des_id", -2))
        if des >= 0 and f"DES_{des}" in V.idx:
            dt = V[f"DES_{des}"]
        elif des == -1 or des >= 0:
            dt = V["DES_OTHER"]
        else:
            dt = V["DES_UNK"]
        ver = int(c.get("versionid", -1))
        vt = V[f"VER_{ver}"] if 0 <= ver < 32 else V["VER_UNK"]
        t = np.array([lvt, slt, typ, yn("TOUCH", c.get("touch")), yn("EXN", c.get("ex")),
                      yn("FEST", c.get("fest")), dt, vt], dtype=np.int32)
        self._cond[ci] = t
        return t

    def _bpm_prog(self, ci, s):
        c, V = self.charts[ci], self.V
        tm = self._tm(ci)
        sb = s / TPB
        bpm = tm.bpm_at(sb)
        bt = V[f"BPM_{min(max(_rnd((bpm - 40) / 10), 0), 39)}"]
        song_beats = float(tm.to_beat(max(float(c["duration"]) - float(c["first"]), 0.0)))
        if not song_beats > 0:
            song_beats = max(int(c["last_tick"]), 1) / TPB
        pr = min(max(int(math.floor(10 * sb / song_beats)), 0), 9)
        return bt, V[f"PROG_{pr}"]

    # ---------------------------------------------------------------- token sequence
    def _events(self, ticks, offs, toks, a, b, s, out_t, out_c):
        if b <= a:
            return
        o0 = int(offs[a])
        tk = toks[o0:int(offs[b])].tolist()
        of = (offs[a:b + 1] - o0).tolist()
        tl = (ticks[a:b] - s).tolist()
        time0 = self.V.time0
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

    @staticmethod
    def _costs(ticks, offs, a, b):
        """tokens per note including its time token if the tick changes (first note of the block always has one)"""
        if b <= a:
            return np.zeros(0, np.int64), np.zeros(0, bool)
        lens = np.diff(offs[a:b + 1]).astype(np.int64)
        tl = ticks[a:b]
        new = np.empty(b - a, bool)
        new[0] = True
        new[1:] = tl[1:] != tl[:-1]
        return lens + new, new

    def build_tokens(self, ci, mode, k, rng=None, train=False):
        """-> dict(tokens, cur_time, loss_mask) or None if the target alone does not fit."""
        c, V = self.charts[ci], self.V
        s = BAR * int(k)
        ticks, offs, toks = self._arrays(c, mode)
        a = int(np.searchsorted(ticks, s - HIST, "left"))
        m = int(np.searchsorted(ticks, s, "left"))
        b = int(np.searchsorted(ticks, s + TGT, "left"))
        cost_t, _ = self._costs(ticks, offs, m, b)
        tgt_len = int(cost_t.sum())
        budget = self.max_len - HEADER - 1 - tgt_len - 1
        if budget < 0:
            return None
        cost_h, new_h = self._costs(ticks, offs, a, m)
        j = 0
        nh = m - a
        if nh:
            suffix = np.cumsum(cost_h[::-1])[::-1]
            ok = suffix <= budget
            j = int(np.argmax(ok)) if ok.any() else nh
            while j < nh and suffix[j] + (0 if new_h[j] else 1) > budget:
                j += 1
        cond = self._cond_tokens(ci).copy()
        if train and rng is not None and self.cond_p > 0:
            drop = rng.random(8) < self.cond_p
            cond[drop] = self._unk[drop]
        bt, pt = self._bpm_prog(ci, s)
        T = [V.BOS] + cond.tolist() + [bt, pt, V.CTX]
        C = [T_MIN] * HEADER
        self._events(ticks, offs, toks, a + j, m, s, T, C)
        tgt_pos = len(T)
        T.append(V.TGT)
        C.append(0)
        self._events(ticks, offs, toks, m, b, s, T, C)
        T.append(V.EOS)
        C.append(C[-1])
        n = len(T)
        mask = np.zeros(n, np.uint8)
        mask[tgt_pos:n - 1] = 1
        return {"tokens": np.asarray(T, np.int32), "cur_time": np.asarray(C, np.int32), "loss_mask": mask}

    # ---------------------------------------------------------------- audio
    def _mel_info(self, sid):
        info = self._mels.get(sid)
        if info is None:
            path = os.path.join(self.mel_dir, f"{int(sid):05d}.npy")
            with open(path, "rb") as fh:
                ver = np.lib.format.read_magic(fh)
                rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dt = rd(fh)
                info = (path, fh.tell(), shape[0], dt)
            self._mels[sid] = info
            if len(self._mels) > 4096:
                self._mels.popitem(last=False)
        return info

    def _read_mel(self, sid, lo, hi):
        """frames [lo, hi) of the song's mel as an in-memory array (one sequential file read)"""
        path, off, T, dt = self._mel_info(sid)
        lo, hi = max(lo, 0), min(hi, T)
        with open(path, "rb") as fh:
            fh.seek(off + lo * N_MEL * dt.itemsize)
            return np.fromfile(fh, dt, count=(hi - lo) * N_MEL).reshape(-1, N_MEL), lo

    def build_mel(self, ci, s, rng=None, train=False):
        """-> float16 [1728, 96], normalized, invalid frames = -4"""
        c = self.charts[ci]
        tm = self._tm(ci)
        ticks = np.arange(s - HIST, s - HIST + N_FRAMES, dtype=np.float64)
        times = float(c["first"]) + tm.to_sec(ticks / TPB)
        if train and rng is not None:
            center = times.mean()
            times = center + (times - center) * (1.0 + rng.normal(0.0, 0.0015)) + rng.uniform(-0.012, 0.012)
        T = self._mel_info(c["sid"])[2]
        f = times * FPS
        lo = max(int(math.floor(f.min())), 0)
        hi = min(int(math.floor(f.max())) + 2, T)
        if hi - lo < 2:
            return np.full((N_FRAMES, N_MEL), INVALID, np.float16)
        sub, lo = self._read_mel(c["sid"], lo, hi)
        x, valid = beat_sync(sub, times - lo / FPS)
        mean = np.asarray(c["mel_mean"], np.float32)
        std = np.maximum(np.asarray(c["mel_std"], np.float32), 1e-3)
        x = (x - mean) / std
        if train and rng is not None:
            x = x + (rng.uniform(-0.3, 0.3) * (np.arange(N_MEL, dtype=np.float32) / 95.0 - 0.5))[None, :]
            if rng.random() < 0.3:
                w = int(rng.integers(1, 9))
                st = int(rng.integers(0, N_MEL - w + 1))
                x[:, st:st + w] = 0.0
        x[~valid] = INVALID
        return x.astype(np.float16)


# ----------------------------------------------------------------------------------------------
class WindowDataset(_Source, IterableDataset):
    """Infinite stream of random training windows (chart ~ #bars, mirror N/H, random bar).

    batch_size=None: yields single samples (dicts of numpy arrays).
    batch_size=B:    yields collated batches (use DataLoader(batch_size=None)).  With bucket=K>1 a pool of K*B
                     samples is sorted by sequence length and cut into K batches (shuffled order), which cuts
                     padding waste a lot.  Every sample is still used once, so the token-level statistics are
                     unchanged; the trainer therefore normalizes the loss by a constant, not by the batch size.
    """

    def __init__(self, data, meta, mel_dir, split="train", seed=0, max_len=MAX_LEN, cond_p=COND_P, augment=True,
                 batch_size=None, bucket=1):
        _Source.__init__(self, data, meta, mel_dir, split, max_len, cond_p)
        self.seed, self.augment = seed, augment
        self.batch_size, self.bucket = batch_size, max(int(bucket), 1)

    def _draw(self, rng, with_mel=True):
        ci = min(int(np.searchsorted(self.cum, rng.random(), "right")), len(self.charts) - 1)
        mode = TRAIN_MODES[int(rng.integers(len(TRAIN_MODES)))]
        k = int(rng.integers(0, int(self.charts[ci]["last_tick"]) // BAR + 1))
        samp = self.build_tokens(ci, mode, k, rng, self.augment)
        if samp is None:
            return None
        if with_mel:
            samp["mel"] = self.build_mel(ci, BAR * k, rng, self.augment)
        return samp

    def mean_loss_tokens(self, n=1000, seed=12345):
        """average number of loss-masked positions per sample (used as the constant loss normalizer)"""
        self._load()
        rng = np.random.default_rng(seed)
        tot, cnt = 0, 0
        while cnt < n:
            smp = self._draw(rng, with_mel=False)
            if smp is not None:
                tot += int(smp["loss_mask"].sum()); cnt += 1
        return tot / cnt

    def _stream(self, rng):
        nerr = 0
        while True:
            try:
                samp = self._draw(rng)
                if samp is None:
                    continue
                nerr = 0
            except Exception:                           # e.g. a missing mel file: skip, but do not loop forever
                nerr += 1
                if nerr > 200:
                    raise
                continue
            yield samp

    def __iter__(self):
        self._load()
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng(np.random.SeedSequence([int(self.seed) & 0xFFFFFFFF, wid]))
        if self.batch_size is None:
            yield from self._stream(rng)
            return
        B, K = self.batch_size, self.bucket
        pool = []
        for samp in self._stream(rng):
            pool.append(samp)
            if len(pool) == B * K:
                if K > 1:
                    pool.sort(key=lambda x: len(x["tokens"]))
                batches = [pool[i:i + B] for i in range(0, len(pool), B)]
                for j in rng.permutation(len(batches)):
                    yield collate(batches[int(j)])
                pool = []


class FixedWindows(_Source, Dataset):
    """Fixed validation set: n windows, mode N, no condition dropout, no augmentation (seed 0)."""

    def __init__(self, data, meta, mel_dir, split="val", n=2000, seed=0, mode="N", max_len=MAX_LEN):
        _Source.__init__(self, data, meta, mel_dir, split, max_len, 0.0)
        self.n, self.seed, self.mode = n, seed, mode
        self.items = None

    def build_items(self):
        self._load()
        rng = np.random.default_rng(self.seed)
        items = []
        tries = 0
        while len(items) < self.n and tries < 50 * self.n:
            tries += 1
            ci = min(int(np.searchsorted(self.cum, rng.random(), "right")), len(self.charts) - 1)
            k = int(rng.integers(0, int(self.charts[ci]["last_tick"]) // BAR + 1))
            if self.build_tokens(ci, self.mode, k) is not None:
                items.append((ci, k))
        self.items = items
        return self

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        self._load()
        ci, k = self.items[i]
        samp = self.build_tokens(ci, self.mode, k)
        samp["mel"] = self.build_mel(ci, BAR * k)
        return samp


def collate(batch, pad_id=0):
    L = max(len(b["tokens"]) for b in batch)
    L = (L + 7) // 8 * 8
    B = len(batch)
    tok = np.full((B, L), pad_id, np.int64)
    ct = np.zeros((B, L), np.int64)
    lm = np.zeros((B, L), bool)
    for i, b in enumerate(batch):
        n = len(b["tokens"])
        tok[i, :n] = b["tokens"]
        ct[i, :n] = b["cur_time"]
        lm[i, :n] = b["loss_mask"]
    mel = np.stack([b["mel"] for b in batch])
    out = {"tokens": tok, "cur_time": ct, "loss_mask": lm, "mel": mel}
    if torch is not None:
        out = {k: torch.from_numpy(v) for k, v in out.items()}
    return out
