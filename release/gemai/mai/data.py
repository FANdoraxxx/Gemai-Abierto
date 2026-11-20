"""Windowed training data built from work/charts.pkl (+ work/meta.json, work/mel/*.npy).  See docs/MODEL_SPEC.md.

Sequence layout:
  BOS LV SLOT TYPE TOUCH EXN FEST DES VER BPM PROG CTX <history events> TGT <target events> EOS
cur_time[i] = tick (relative to the window start s) of the latest time token at/before position i
(-768 before the first history time token, 0 at TGT).  loss_mask[i] = 1 where tokens[i+1] is a target token / EOS.

Training sampling: chart ~ bars * SLOT_W[slot] (MASTER / Re:MASTER are what the user wants -> x3), mirror mode uniform
over N/H (never V/HV), bar uniform.  Validation: fixed windows from slots 5/6 of the val split only.

Speed notes (the box is a 4-core i3 with a slow disk and an unreliable OS file cache):
  * all mel files live in ONE shared-memory block (create_mel_shm, filled once by the trainer process, attached by name in
    every worker / the val builder) -> no per-sample file IO, no per-worker copies.  Without it, per-file np.load mmap is used.
  * per-chart constants (NpTempo, condition tokens, mel gain, song length) are computed once in _load.
  * build_mel gathers the two neighbouring frames, and does interpolation + normalization + spectral tilt in one fused pass.
"""
from __future__ import annotations
import os, json, pickle, time
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
from .audio import FPS
from .sequence import (BAR, HIST, TGT, FUT_TICKS, LOOK, N_FRAMES, N_MEL, MAX_LEN, HEADER, INVALID,
                       condition_tokens, bpm_progress_tokens, build_window_tokens, window_mel, window_times, unk_condition_row,
                       density_token, plan_token, sec_token, secn_token)
from . import v2 as V2
from . import v3 as V3
from . import ann as A
from . import feats as FE
FAN_MEL_FALLBACK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "work_fan2", "mel")   # G: originals of the fan mel

MODES = ("N", "H", "V", "HV")      # all modes present in charts.pkl
TRAIN_MODES = ("N", "H")           # only left-right mirroring is used for training (V/HV break playing habits)
SLOT_W = {2: 0.5, 3: 0.5, 4: 1.0, 5: 3.0, 6: 3.0}   # sampling weight per bar by difficulty slot (5=MASTER, 6=Re:MASTER)
VAL_SLOTS = (5, 6)
COND_P = 0.3
DEFAULT_V2_OPTS = dict(hist_corrupt=True, p_hist_drop_all=0.12, p_hist_thin=0.22, p_hist_strip=0.16, p_hist_note=0.10,
                       audio_aug=True,
                       p_dens=0.5, p_all_unk=0.05, p_amt_y=0.25, dens_mix=0.0)


DEFAULT_V3_OPTS = dict(song=True,            # sample carries the whole-song bar table (+ aux targets)
                       ref=True,             # REF segment (retrieval of an earlier similar section); False -> always REF_NONE
                       p_plan=0.5,           # PLAN token -> PLAN_UNK dropout (train)
                       p_ref_drop=0.15,      # REF segment dropped (REF_NONE) (train)
                       p_ref_note=0.08,      # per-note dropout inside the REF segment (train)
                       ref_budget=None, ref_lite=None)    # None: values of v3_stats.json


DEFAULT_V4_OPTS = dict(ann=True,             # use the LLM structure annotations (work/llm_annot/v4_ann.pkl); False -> every song is UNK
                       p_ann=0.4,            # train: whole annotation channel (song bars + window tokens) -> UNK
                       p_win=0.15,           # train, channel on: window tokens only -> UNK
                       p_bars=0.10,          # train, channel on: per-bar song inputs only -> UNK
                       p_label=0.10, p_jit=0.10, p_mom=0.10,     # train: label noise per section / boundary +-1 bar / dropped moments (mai.ann.noisy)
                       val_ann="full")       # validation windows: "full" | "off" | "bars" | "win" (which channels are shown)


def specaug(mel, f, rng, n_t=2, max_t=48, n_b=2, max_b=8):
    """SpecAugment-style input masking of one training window (v6).  mel float16 [1728, 96] (tick grid, 48 frames = 1 beat, invalid frames == -4),
    f = dict(fsl, fi0, fw, fvalid) of the MERT window on the same grid (or None).  n_t time spans of 1..max_t frames are set to 0 (the normalized mean) in the
    valid part of the mel and their MERT frames become invalid (the model then uses its learned 'no feature' vector); n_b bands of 1..max_b mel bins are zeroed.
    Returns (new mel, f) -- f['fvalid'] is replaced by a copy, the inputs are not modified."""
    y = np.array(mel, np.float16)
    valid = (y[:, 0] != INVALID) | (y[:, y.shape[1] // 2] != INVALID)
    fv = None if f is None else np.array(f["fvalid"], bool)
    for _ in range(n_t):
        w = int(rng.integers(1, max_t + 1)); s = int(rng.integers(0, y.shape[0] - w + 1))
        seg = valid[s:s + w]
        y[s:s + w][seg] = 0
        if fv is not None:
            fv[s:s + w] = False
    for _ in range(n_b):
        w = int(rng.integers(1, max_b + 1)); b0 = int(rng.integers(0, y.shape[1] - w + 1))
        y[valid, b0:b0 + w] = 0
    if f is not None:
        f = dict(f); f["fvalid"] = fv
    return y, f


def parse_slot_w(s):
    """'2:0.5,3:0.5,4:1,5:3,6:3' -> dict"""
    return {int(k): float(v) for k, v in (kv.split(":") for kv in s.split(",") if kv)}


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


def make_vocab(meta, seq_version=1, with_fut=False):
    return Vocab(designer_names(meta), seq_version=seq_version, with_fut=with_fut)


# ----------------------------------------------------------------------------------------------
# shared-memory mel store
def _npy_header(path):
    with open(path, "rb") as fh:
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dt = rd(fh)
        return fh.tell(), shape, dt


def create_mel_shm(mel_dir, log=print, threads=3):
    """Read every mel/*.npy (float16 [T,96]) into ONE shared-memory block.  Returns (shm, desc).
    Keep `shm` referenced as long as the data is needed (the block disappears with its last handle); `desc` is small and
    picklable -> pass it as mel_shm= to WindowDataset / FixedWindows, which attach by name in every process."""
    from multiprocessing import shared_memory
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.perf_counter()
    files = sorted(f for f in os.listdir(mel_dir) if f.endswith(".npy"))
    ent, index, total = [], {}, 0
    for f in files:
        off, shape, dt = _npy_header(os.path.join(mel_dir, f))
        assert dt == np.float16 and len(shape) == 2 and shape[1] == N_MEL, (f, shape, dt)
        index[int(f[:-4])] = (total, int(shape[0]))
        ent.append((f, off, total, int(shape[0])))
        total += int(shape[0])
    nbytes = total * N_MEL * 2
    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
    raw = np.ndarray((nbytes,), np.uint8, buffer=shm.buf)

    def load(e):
        f, off, fo, T = e
        with open(os.path.join(mel_dir, f), "rb", buffering=0) as fh:
            fh.seek(off)
            mv = memoryview(raw[fo * N_MEL * 2:(fo + T) * N_MEL * 2])
            got = 0
            while got < len(mv):
                n = fh.readinto(mv[got:])
                if not n:
                    raise IOError("short read " + f)
                got += n
        return len(mv)

    with ThreadPoolExecutor(threads) as ex:
        done = sum(ex.map(load, ent))
    dt = time.perf_counter() - t0
    log(f"[mel] {len(files)} files, {done/1e9:.2f} GB loaded into shared memory in {dt:.0f}s ({done/1e6/max(dt,1e-9):.0f} MB/s)")
    return shm, {"name": shm.name, "total": total, "index": index}


# ----------------------------------------------------------------------------------------------
class _Source:
    """Lazily loaded chart store.  The big pickle is loaded inside each worker process (never pickled across)."""

    def __init__(self, data, meta, mel_dir, split, max_len=MAX_LEN, cond_p=COND_P, slots=None, slot_w=None,
                 keep_modes=TRAIN_MODES, mel_shm=None, seq_version=1, v2_stats=None, v2_opts=None, feats=None,
                 v3_stats=None, v3_opts=None, bars_root=None, v4_opts=None, ann_path=None, era_w=1.0, era_cut=10,
                 extra_data=None, extra_mel=None, extra_v2_stats=None, fan_mult=1.0, specaug_p=0.0,
                 with_fut=False, p_fut=0.0):
        # v6 infill: with_fut adds the FUT token to the vocabulary; p_fut = share of TRAIN samples using the infill layout
        self.with_fut = bool(with_fut)
        self.p_fut = float(p_fut)
        # v6: specaug_p > 0 -> with this probability a TRAIN window gets 2 time spans (<= 1 beat) and 2 mel bands (<= 8 bins) zeroed; the same time spans make the MERT frames invalid
        self.specaug_p = float(specaug_p)
        # v5: extra_data = charts.pkl of a second dataset (fan charts, sid >= feats.FAN_SID0) added to the TRAIN stream only (never validation);
        # extra_mel / extra_v2_stats: its mel dir / v2 stats table; each extra chart is sampled with weight nbars * slot_w * c['weight'] * fan_mult
        # (no era weighting for them).  Without extra_data this class behaves exactly like the v4 one.
        self.extra_data = str(extra_data) if extra_data else None
        self.extra_mel = str(extra_mel) if extra_mel else None
        self.extra_v2_stats = str(extra_v2_stats) if extra_v2_stats else None
        self.fan_mult = float(fan_mult)
        if self.extra_data and mel_shm is not None:
            raise ValueError("extra_data is not supported together with the shared-memory mel store (use --no_shm)")
        # v4: LLM structure annotations (mai/ann.py) -> per-bar song-encoder inputs + SEC/SECN window tokens; era_w: sampling weight factor of the
        # charts with versionid < era_cut (pre-MiLK) in the training stream
        self.v4o = dict(DEFAULT_V4_OPTS)
        self.v4o.update(v4_opts or {})
        self.ann_path = str(ann_path) if ann_path else os.path.join(os.path.dirname(str(meta)) or ".", "llm_annot", "v4_ann.pkl")
        self.era_w, self.era_cut = float(era_w), int(era_cut)
        # feats: None (mel only) or dict(encoder=, layer=, dim=, fps=[, root=]) -> samples also carry the pretrained-feature window
        self.feats = dict(feats) if feats else None
        # seq_version 3 (mai/v3.py): PLAN token, REF segment and the whole-song bar table.  v3_stats: work/v3_stats.json (plan edges /
        # ref thresholds); bars_root: directory of the per-chart bar tables (default $MAI_BARS_ROOT or $MAI_FEATS_ROOT/bars_v3)
        self.v3_stats_path = str(v3_stats) if v3_stats else os.path.join(os.path.dirname(str(meta)) or ".", "v3_stats.json")
        self.v3o = dict(DEFAULT_V3_OPTS)
        self.v3o.update(v3_opts or {})
        self.bars_root = str(bars_root) if bars_root else None
        self.data, self.meta, self.mel_dir = str(data), str(meta), str(mel_dir)
        # seq_version 2 (see mai/v2.py): amount / density conditioning, history corruption, stronger audio aug (train only)
        self.seq_version = int(seq_version)
        self.v2_stats = str(v2_stats) if v2_stats else os.path.join(os.path.dirname(str(meta)) or ".", "v2_stats.json")
        self.v2o = dict(DEFAULT_V2_OPTS)
        self.v2o.update(v2_opts or {})
        self.split, self.max_len, self.cond_p = split, max_len, cond_p
        self.slots = None if slots is None else tuple(int(x) for x in slots)
        self.slot_w = None if slot_w is None else dict(slot_w)
        self.keep_modes = tuple(keep_modes)
        self.mel_shm = mel_shm
        self.L = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["L"] = None
        return d

    # attribute access for the lazily loaded parts (kept as plain attributes for readability / old tests)
    @property
    def charts(self): self._load(); return self.L.charts
    @property
    def V(self): self._load(); return self.L.V
    @property
    def cum(self): self._load(); return self.L.cum
    @property
    def nbars(self): self._load(); return self.L.nbars

    # ---------------------------------------------------------------- loading
    def _load(self):
        if self.L is not None:
            return
        class _NS:  # simple namespace
            pass
        L = _NS()
        with open(self.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        L.V = V = make_vocab(meta, self.seq_version, with_fut=self.with_fut or self.p_fut > 0)
        with open(self.data, "rb") as f:
            allc = pickle.load(f)
        ch = [c for c in allc if c.get("split") == self.split and (self.slots is None or int(c["slot"]) in self.slots)]
        del allc
        if self.extra_data and self.split == "train":
            with open(self.extra_data, "rb") as f:
                exc = pickle.load(f)
            exc = [c for c in exc if c.get("split", "train") == "train" and (self.slots is None or int(c["slot"]) in self.slots)]
            assert all(int(c["sid"]) < FE.FAN_SID0 for c in ch), "official sids must be below the fan id space"
            assert all(int(c["sid"]) >= FE.FAN_SID0 for c in exc), "extra-data sids must be >= FAN_SID0"
            ch = ch + exc
            del exc
        for c in ch:                                         # drop the mirror modes we never use (memory)
            c["modes"] = {m: c["modes"][m] for m in self.keep_modes}
        L.charts = ch
        L.nbars = np.array([int(c["last_tick"]) // BAR + 1 for c in ch], dtype=np.float64)
        w = L.nbars.copy()
        if self.slot_w is not None:
            w *= np.array([self.slot_w.get(int(c["slot"]), 1.0) for c in ch], dtype=np.float64)
        L.is_fan = np.array([int(c["sid"]) >= FE.FAN_SID0 for c in ch], dtype=bool)
        if self.era_w != 1.0:
            w *= np.array([self.era_w if (int(c.get("versionid", 99)) < self.era_cut and int(c["sid"]) < FE.FAN_SID0) else 1.0 for c in ch], dtype=np.float64)
        w_off = np.where(L.is_fan, 0.0, w)                   # official-only stream (loss normalizer, reporting)
        if L.is_fan.any():
            w = np.where(L.is_fan, w * np.array([float(c.get("weight", 1.0)) for c in ch], dtype=np.float64) * self.fan_mult, w)
        L.mass_off, L.mass_fan = float(w_off.sum()), float(w[L.is_fan].sum())
        L.cum = np.cumsum(w)
        L.cum /= L.cum[-1]
        L.cum_off = np.cumsum(w_off)
        L.cum_off /= L.cum_off[-1]
        L.unk = np.array([V["LV_UNK"], V["SLOT_UNK"], V["TYPE_UNK"], V["TOUCH_UNK"], V["EXN_UNK"],
                          V["FEST_UNK"], V["DES_UNK"], V["VER_UNK"]], dtype=np.int32)
        L.v3 = V3.load_v3_stats(self.v3_stats_path) if self.seq_version >= 3 else None
        L.cnt, L.ratio = {}, {}
        if self.seq_version >= 2:
            L.unk = unk_condition_row(V)
            st = V2.load_stats(self.v2_stats)
            L.v2meta = V2.seq_meta_of(st)
            tab = st["charts"]
            if L.is_fan.any():                               # fan charts: own table, same thresholds / density edges as the official one
                ex_st = V2.load_stats(self.extra_v2_stats)
                assert ex_st["dens_edges"] == st["dens_edges"] and ex_st["thr"] == st["thr"], "extra v2 stats use other bucket edges"
                assert not (set(ex_st["charts"]) & set(tab)), "v2 stats key collision"
                tab = {**tab, **ex_st["charts"]}
            for c in ch:                                     # touch_b / ex_b / fest_b / dens_b from the stats table
                r = tab[f"{int(c['sid'])}:{int(c['slot'])}"]
                V2.annotate_chart(c, dict(nn=r[0], nt=r[1], ne=r[2], nf=r[3], dens=r[4]), L.v2meta)
                c["nn_"], c["dens_s"] = int(r[0]), float(r[4])
        self.L = L                                           # (cond / tempo helpers below use self.L.V etc.)
        L.tm, L.first, L.sid, L.cond, L.song_beats, L.inv, L.negmi, L.nbars_audio = [], [], [], [], [], [], [], []
        for ci, c in enumerate(ch):
            tm = NpTempo(c["beats"], c["bpms"])
            L.tm.append(tm)
            L.first.append(float(c["first"]))
            L.sid.append(int(c["sid"]))
            L.cond.append(self._cond_tokens(c))
            L.nbars_audio.append(V3.n_bars_of(tm, c["first"], c["duration"], c["last_tick"]) if self.seq_version >= 3 else 0)
            sb = float(tm.to_beat(max(float(c["duration"]) - float(c["first"]), 0.0)))
            L.song_beats.append(sb if sb > 0 else max(int(c["last_tick"]), 1) / TPB)
            std = np.maximum(np.asarray(c["mel_std"], np.float32), 1e-3).reshape(1, -1)
            mean = np.asarray(c["mel_mean"], np.float32).reshape(1, -1)
            inv = (1.0 / std).astype(np.float32)
            L.inv.append(inv)
            L.negmi.append((-mean * inv).astype(np.float32))
        L.ann = [None] * len(ch)                              # AnnRecord per chart (None = no annotation -> UNK); loaded for every seq_version
        if self.v4o["ann"]:                                   # (only seq_version >= 4 uses it for the samples; older ones just report coverage)
            store = A.load_store(self.ann_path)
            for ci, c in enumerate(ch):
                d = store.get((int(c["sid"]), int(c["slot"])))
                if d is not None:
                    r = A.AnnRecord.from_dict(d)
                    if r.nb == L.nbars_audio[ci] or self.seq_version < 3:
                        L.ann[ci] = r
        # mel access
        L.shm = None
        L.mel_all = None
        L.mm = {}
        L.fm = {}
        if self.mel_shm is not None:
            from multiprocessing import shared_memory
            L.shm = shared_memory.SharedMemory(name=self.mel_shm["name"])
            L.mel_all = np.ndarray((self.mel_shm["total"], N_MEL), np.float16, buffer=L.shm.buf)
            L.index = self.mel_shm["index"]

    # ---------------------------------------------------------------- per-chart helpers
    @staticmethod
    def _arrays(c, mode):
        d = c["modes"][mode]
        return d["ticks"], d["offs"], d["toks"]

    def _cond_tokens(self, c):
        return condition_tokens(self.L.V, c)

    def _chart_arrays(self, ci):
        c = self.L.charts[ci]
        return self._arrays(c, next(iter(c["modes"])))              # per-bar note statistics do not depend on the (left/right) mirror

    def bar_counts(self, ci):
        """float32 [nb, 4] per-bar notes / hold-like / slide / touch-like of the chart (cached)"""
        L = self.L
        r = L.cnt.get(ci)
        if r is None:
            t, o, k = self._chart_arrays(ci)
            r = L.cnt[ci] = V3.bar_counts(L.V, t, o, k, L.nbars_audio[ci])
        return r

    def plan_ratio(self, ci):
        """window (4 bars from k) notes per bar / song mean notes per bar, for every start bar k (cached)"""
        L = self.L
        r = L.ratio.get(ci)
        if r is None:
            r = L.ratio[ci] = V3.window_ratio(self.bar_counts(ci)[:, 0])
        return r

    def bar_table(self, ci):
        c = self.L.charts[ci]
        return V3.load_bar_table(c["sid"], c["slot"], self.bars_root)

    def _plan_ref(self, ci, mode, k, ticks, offs, toks, rng, train):
        """-> (PLAN token, (REF marker, ref event tokens, ref cur_time) )  [ground truth plan; reference from the chart's own earlier notes]"""
        L = self.L
        V = L.V
        st = L.v3
        pr = self.plan_ratio(ci)
        pb = int(V3.plan_bucket(pr[min(int(k), len(pr) - 1)], st["plan_edges"]))
        ref = (V["REF_NONE"], [], [])
        if self.v3o["ref"] and not (train and rng is not None and rng.random() < self.v3o["p_ref_drop"]):
            tab = self.bar_table(ci)
            if tab is not None:
                bud = self.v3o["ref_budget"] if self.v3o["ref_budget"] is not None else st["ref_budget"]
                lite = self.v3o["ref_lite"] if self.v3o["ref_lite"] is not None else st["ref_lite"]
                mk, t, c, _ = V3.retrieve_ref(V, V3.bar_embed(tab), ticks, offs, toks, int(k), st["ref_edges"], bud, lite,
                                              rng if train else None, self.v3o["p_ref_note"] if train else 0.0)
                ref = (mk, t, c)
        return plan_token(V, pb), ref

    def mass_report(self):
        """sampling mass of the official / fan charts: dict(n_off, n_fan, mass_off, mass_fan, fan_share) -- fan_share = probability that a drawn window is a fan window"""
        self._load()
        L = self.L
        tot = L.mass_off + L.mass_fan
        return dict(n_off=int((~L.is_fan).sum()), n_fan=int(L.is_fan.sum()), mass_off=L.mass_off, mass_fan=L.mass_fan, fan_share=L.mass_fan / tot if tot > 0 else 0.0)

    def make_ann(self, ci, rng=None, train=False, mode=None):
        """annotation state of one sample: None (chart without annotation, or channel dropped) | dict(arr=per-bar arrays, win=bool, bars=bool)"""
        rec = self.L.ann[ci]
        if rec is None:
            return None
        o = self.v4o
        if train and rng is not None:
            if rng.random() < o["p_ann"]:
                return None
            win, bars = rng.random() >= o["p_win"], rng.random() >= o["p_bars"]
            rec = A.noisy(rec, rng, o["p_label"], o["p_jit"], o["p_mom"])
        else:
            mode = mode or o["val_ann"]
            win, bars = mode in ("full", "win"), mode in ("full", "bars")
        if not (win or bars):
            return None
        return dict(arr=A.bar_arrays(rec), win=win, bars=bars)

    def build_song(self, ci, k, ann=None):
        """whole-song arrays for the song encoder + auxiliary targets (see mai.v3.song_arrays); a chart without bar table gets one dummy bar
        with smask False (the model switches the song path off for it)"""
        L = self.L
        tab = self.bar_table(ci)
        if tab is None:
            out = dict(sfeat=np.zeros((1, V3.BAR_COLS), np.float16), sscal=np.zeros((1, 4), np.float32), sidx=np.zeros(1, np.int64),
                       srel=np.full(1, V3.REL_HALF, np.int64), smask=np.zeros(1, bool), saux=np.zeros((1, V3.AUX_COLS), np.float32),
                       snb=1, sg=1, sk=int(k))
            if self.seq_version >= 4:
                out.update(alab=np.zeros(1, np.int64), apos=np.zeros(1, np.int64), aend=np.zeros(1, np.int64),
                           aflg=np.zeros((1, A.N_FLG), np.float32))
            return out
        tab = np.asarray(tab)
        out = V3.song_arrays(tab, L.tm[ci], k)
        nb = len(tab)
        g = V3.pool_factor(nb)
        cnt = self.bar_counts(ci)
        if len(cnt) != nb:                                              # table built from a different bar count: align (should not happen)
            cnt = np.pad(cnt, ((0, max(nb - len(cnt), 0)), (0, 0)))[:nb]
        out["saux"] = V3.aux_targets(cnt, nb, g)
        out["smask"] = np.ones(len(out["sfeat"]), bool)
        out["snb"], out["sg"], out["sk"] = nb, g, int(k)
        if self.seq_version >= 4:                                       # annotation channel of the song encoder (UNK rows = zeros)
            arr = ann["arr"] if (ann is not None and ann["bars"] and len(ann["arr"]["lab"]) == nb) else A.unk_arrays(nb)
            p = A.pool_arrays(arr, g)
            out.update(alab=p["lab"], apos=p["pos"], aend=p["end"], aflg=p["flg"])
        return out

    def _bpm_prog(self, ci, s):
        L = self.L
        return bpm_progress_tokens(L.V, L.tm[ci], s, L.song_beats[ci])

    # ---------------------------------------------------------------- token sequence
    def build_tokens(self, ci, mode, k, rng=None, train=False, ann=None, with_fut=None,
                     fut_ci=None, fut_k=None):
        """-> dict(tokens, cur_time, loss_mask) or None if the target alone does not fit.

        ``with_fut``: use the v6 infill layout (FUT + the true future 8 beats + TGT).  ``fut_ci`` / ``fut_k``:
        take the future block from another chart / bar (cyclic-shifted wrong future for the 3-way validation)."""
        self._load()
        L = self.L
        s = BAR * int(k)
        ticks, offs, toks = self._arrays(L.charts[ci], mode)
        ticks_orig = ticks
        bt, pt = self._bpm_prog(ci, s)
        ticks_full, offs_full, toks_full = ticks, offs, toks          # uncorrupted arrays of the whole chart (REF source)
        conds, ref = L.cond[ci], None
        if self.seq_version >= 3:
            ptok, ref = self._plan_ref(ci, mode, k, ticks_full, offs_full, toks_full, rng, train)
            conds = conds.copy()
            conds[L.V.plan_col - 1] = ptok
            if self.seq_version >= 4:                                   # v4: section label of the first bar + "new section inside the window"
                sec = secn = None
                if ann is not None and ann["win"]:
                    sec, secn = A.window_tokens(ann["arr"], int(k))
                conds[L.V.sec_col - 1] = sec_token(L.V, sec)
                conds[L.V.secn_col - 1] = secn_token(L.V, secn)
        if self.seq_version >= 2 and train and rng is not None and self.v2o["hist_corrupt"]:
            cw = V2.corrupt_window_arrays(L.V, ticks, offs, toks, s, rng, p_drop_all=self.v2o["p_hist_drop_all"],
                                          p_thin=self.v2o["p_hist_thin"], p_strip=self.v2o["p_hist_strip"],
                                          p_note=self.v2o["p_hist_note"])
            if cw is not None:
                ticks, offs, toks = cw
        if with_fut is None:
            with_fut = self.with_fut
        fut_arr, fut_start, n_fut = (ticks_full, offs_full, toks_full), s + TGT, 0
        if with_fut:
            if fut_ci is not None:
                ft, fo, fk = self._arrays(L.charts[int(fut_ci)], mode)
                fut_arr, fut_start = (ft, fo, fk), BAR * int(fut_k) + TGT
            n_fut = int(np.searchsorted(fut_arr[0], fut_start + FUT_TICKS, "left")
                        - np.searchsorted(fut_arr[0], fut_start, "left"))
        sample = build_window_tokens(L.V, ticks, offs, toks, s, conds, bt, pt,
                                     max_len=self.max_len, ref=ref, with_fut=with_fut,
                                     fut=fut_arr, fut_start=fut_start)
        if sample is None:
            return None
        sample["n_fut"] = int(n_fut) if with_fut else 0
        if self.seq_version >= 2 and train and rng is not None and self.v2o["dens_mix"] > 0:
            self._mix_window_density(sample, ci, ticks_orig, s)
        if train and rng is not None and self.cond_p > 0:
            if self.seq_version < 2:
                drop = rng.random(8) < self.cond_p
                sample["tokens"][1:9][drop] = L.unk[drop]
            else:
                self._cond_dropout_v2(sample["tokens"], rng)
        return sample

    def _mix_window_density(self, sample, ci, ticks, s):
        """dens_mix w > 0: the DENS token carries a blend of chart-level and *this window's* density,
        bucket(chart_dens * r**w), r = window notes per beat / chart notes per beat (clipped 0.2..5).
        Gives the token a strong, learnable meaning; with w = 0 it is the pure chart-level bucket."""
        L = self.L
        c = L.charts[ci]
        n_t = int(np.searchsorted(ticks, s + TGT, "left") - np.searchsorted(ticks, s, "left"))
        beats_t = TGT / TPB
        chart_pb = max(c["nn_"], 1) / max(float(L.song_beats[ci]), 1.0)
        r = min(max((n_t / beats_t) / chart_pb, 0.2), 5.0)
        d = c["dens_s"] * (r ** float(self.v2o["dens_mix"]))
        b = V2.density_bucket(d, L.v2meta.get("dens_edges", V2.DENS_EDGES))
        sample["tokens"][L.V.dens_col] = density_token(L.V, b)

    def _cond_dropout_v2(self, tokens, rng):
        """9 conditions: each -> UNK with p=cond_p (DENS: p_dens); with p_all everything -> UNK (CFG row);
        a present amount bucket (L/M/H) becomes the amount-unspecified *_Y token with p_y."""
        V = self.L.V
        n = V.n_cond
        cond = tokens[1:1 + n]
        if rng.random() < self.v2o["p_all_unk"]:
            cond[:] = self.L.unk
            return
        p = np.full(n, self.cond_p)
        p[V.dens_col - 1] = self.v2o["p_dens"]
        if V.seq_version >= 3:
            p[V.plan_col - 1] = self.v3o["p_plan"]
        if V.seq_version >= 4:
            p[V.sec_col - 1] = p[V.secn_col - 1] = 0.0                  # the annotation channel has its own dropout (make_ann)
        drop = rng.random(n) < p
        ysw = rng.random(3) < self.v2o["p_amt_y"]
        for j, name in enumerate(("TOUCH", "EXN", "FEST")):
            if ysw[j] and cond[3 + j] in (V[name + "_L"], V[name + "_M"], V[name + "_H"]):
                cond[3 + j] = V[name + "_Y"]
        cond[drop] = self.L.unk[drop]

    # ---------------------------------------------------------------- audio
    def _mel_of(self, sid):
        """song mel as a float16 [T,96] array (view into shared memory, or a read-only np.load mmap)"""
        L = self.L
        if L.mel_all is not None:
            off, T = L.index[sid]
            return L.mel_all[off:off + T]
        mm = L.mm.get(sid)
        if mm is None:
            d_ = self.extra_mel if (int(sid) >= FE.FAN_SID0 and self.extra_mel) else self.mel_dir
            f_ = os.path.join(d_, f"{int(sid):05d}.npy")
            if not os.path.exists(f_) and int(sid) >= FE.FAN_SID0:   # partial SSD cache -> the G: original
                f_ = os.path.join(FAN_MEL_FALLBACK, f"{int(sid):05d}.npy")
            mm = L.mm[sid] = np.asarray(np.load(f_, mmap_mode="r"))
        return mm

    def build_mel(self, ci, s, rng=None, train=False, with_feats=False):
        """-> float16 [1728, 96], normalized, invalid frames = -4.
        with_feats=True and feature fusion configured: -> (mel, dict(fsl, fi0, fw, fvalid)) (same time grid as the mel)"""
        self._load()
        L = self.L
        ot = [] if (with_feats and self.feats) else None
        x = window_mel(self._mel_of(L.sid[ci]), L.tm[ci], L.first[ci], s,
                       None, None, rng=rng, train=train, inv_std=L.inv[ci],
                       neg_mean_std=L.negmi[ci], out_times=ot)
        if self.seq_version >= 2 and train and rng is not None and self.v2o["audio_aug"]:
            x = V2.augment_mel_v2(x, rng)
        if ot is not None:
            return x, self._feat_window(L.sid[ci], ot[0])
        return x

    def _feat_window(self, sid, times):
        """fp16 slice of the frozen-encoder features + interpolation indices for the window time grid (see mai.feats)"""
        L = self.L
        e = self.feats
        F = L.fm.get(sid)
        if F is None:
            try:
                F = np.load(FE.feats_path(e["encoder"], e["layer"], int(sid), e.get("root")), mmap_mode="r")
                if F.ndim != 2 or F.shape[0] < 2 or F.shape[1] != e["dim"]:
                    F = False
            except Exception:
                F = False
            L.fm[sid] = F
        if F is False:                                       # no features for this song: all frames invalid (-> learned "no feature" vector)
            n = len(times)
            return dict(fsl=np.zeros((2, e["dim"]), np.float16), fi0=np.zeros(n, np.int32), fw=np.zeros(n, np.float32),
                        fvalid=np.zeros(n, bool))
        sl, i0, w, valid = FE.window_slice_feats(F, e["fps"], times)
        return dict(fsl=sl, fi0=i0, fw=w, fvalid=valid)


# ----------------------------------------------------------------------------------------------
class WindowDataset(_Source, IterableDataset):
    """Infinite stream of random training windows (chart ~ #bars * slot weight, mirror N/H, random bar).

    batch_size=None: yields single samples (dicts of numpy arrays).
    batch_size=B:    yields collated batches (use DataLoader(batch_size=None)).  With bucket=K>1 a pool of K*B
                     samples is sorted by sequence length and cut into K batches (shuffled order), which cuts
                     padding waste a lot.  Every sample is still used once, so the token-level statistics are
                     unchanged; the trainer therefore normalizes the loss by a constant, not by the batch size.
    """

    def __init__(self, data, meta, mel_dir, split="train", seed=0, max_len=MAX_LEN, cond_p=COND_P, augment=True,
                 batch_size=None, bucket=1, slot_w=SLOT_W, slots=None, keep_modes=TRAIN_MODES, mel_shm=None,
                 seq_version=1, v2_stats=None, v2_opts=None, feats=None, v3_stats=None, v3_opts=None, bars_root=None,
                 v4_opts=None, ann_path=None, era_w=1.0, era_cut=10, extra_data=None, extra_mel=None, extra_v2_stats=None,
                 fan_mult=1.0, specaug_p=0.0, p_fut=0.0):
        _Source.__init__(self, data, meta, mel_dir, split, max_len, cond_p, slots=slots, slot_w=slot_w,
                         keep_modes=keep_modes, mel_shm=mel_shm, seq_version=seq_version, v2_stats=v2_stats,
                         v2_opts=v2_opts, feats=feats, v3_stats=v3_stats, v3_opts=v3_opts, bars_root=bars_root,
                         v4_opts=v4_opts, ann_path=ann_path, era_w=era_w, era_cut=era_cut,
                         extra_data=extra_data, extra_mel=extra_mel, extra_v2_stats=extra_v2_stats, fan_mult=fan_mult,
                         specaug_p=specaug_p, with_fut=p_fut > 0, p_fut=p_fut)
        self.seed, self.augment = seed, augment
        self.batch_size, self.bucket = batch_size, max(int(bucket), 1)

    def _draw(self, rng, with_mel=True, only_official=False):
        self._load()
        L = self.L
        samp = None
        for _try in range(8):
            ci = min(int(np.searchsorted(L.cum_off if only_official else L.cum, rng.random(), "right")), len(L.charts) - 1)
            mode = TRAIN_MODES[int(rng.integers(len(TRAIN_MODES)))]
            with_fut = self._choose_fut(rng)
            last = int(L.charts[ci]["last_tick"])
            if with_fut:                          # prefer windows that actually have a 2-bar future
                kmax = max(1, (last - TGT - FUT_TICKS) // BAR + 1)
                k = int(rng.integers(0, kmax))
            else:
                k = int(rng.integers(0, last // BAR + 1))
            ann = self.make_ann(ci, rng, self.augment) if self.seq_version >= 4 else None
            samp = self.build_tokens(ci, mode, k, rng, self.augment, ann=ann, with_fut=with_fut)
            if samp is not None and (not with_fut or samp.get("n_fut", 0) > 0):
                break
            samp = None
        if samp is None:
            return None
        samp["is_fan"] = bool(L.is_fan[ci])
        if with_mel:
            if self.feats:
                samp["mel"], f = self.build_mel(ci, BAR * k, rng, self.augment, with_feats=True)
                if self.augment and self.specaug_p > 0 and rng.random() < self.specaug_p:
                    samp["mel"], f = specaug(samp["mel"], f, rng)
                samp.update(f)
            else:
                samp["mel"] = self.build_mel(ci, BAR * k, rng, self.augment)
                if self.augment and self.specaug_p > 0 and rng.random() < self.specaug_p:
                    samp["mel"], _ = specaug(samp["mel"], None, rng)
            if self.seq_version >= 3 and self.v3o["song"]:
                samp.update(self.build_song(ci, k, ann=ann))
        return samp

    def _choose_fut(self, rng):
        """v6 infill: True for the infill layout with probability p_fut (0.5 in the V6 infill training)."""
        return bool(self.p_fut > 0 and rng.random() < self.p_fut)

    def mean_loss_tokens(self, n=1000, seed=12345):
        """average number of loss-masked positions per sample (used as the constant loss normalizer)"""
        self._load()
        rng = np.random.default_rng(seed)
        tot, cnt = 0, 0
        while cnt < n:
            smp = self._draw(rng, with_mel=False, only_official=True)       # v5: the normalizer is the official-only stream (same scale as v4)
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
    """Fixed validation set: n windows, mode N, no condition dropout, no augmentation (seed 0).
    Only charts of `slots` (default MASTER / Re:MASTER) of the val split; chart ~ #bars, bar uniform."""

    def __init__(self, data, meta, mel_dir, split="val", n=2000, seed=0, mode="N", max_len=MAX_LEN, slots=VAL_SLOTS,
                 mel_shm=None, seq_version=1, v2_stats=None, feats=None, v3_stats=None, v3_opts=None, bars_root=None,
                 v4_opts=None, ann_path=None, with_fut=False):
        _Source.__init__(self, data, meta, mel_dir, split, max_len, 0.0, slots=slots, slot_w=None,
                         keep_modes=(mode,), mel_shm=mel_shm, seq_version=seq_version, v2_stats=v2_stats, feats=feats,
                         v3_stats=v3_stats, v3_opts=v3_opts, bars_root=bars_root, v4_opts=v4_opts, ann_path=ann_path,
                         with_fut=with_fut)
        self.n, self.seed, self.mode = n, seed, mode
        self.items = None

    def build_items(self):
        self._load()
        L = self.L
        rng = np.random.default_rng(self.seed)
        items = []
        tries = 0
        while len(items) < self.n and tries < 50 * self.n:
            tries += 1
            ci = min(int(np.searchsorted(L.cum, rng.random(), "right")), len(L.charts) - 1)
            k = int(rng.integers(0, int(L.charts[ci]["last_tick"]) // BAR + 1))
            if self.build_tokens(ci, self.mode, k) is not None:
                items.append((ci, k))
        self.items = items
        return self

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        self._load()
        ci, k = self.items[i]
        ann = self.make_ann(ci) if self.seq_version >= 4 else None
        samp = self.build_tokens(ci, self.mode, k, ann=ann)
        samp["mel"] = self.build_mel(ci, BAR * k)
        samp["slot"] = int(self.L.charts[ci]["slot"])
        if self.seq_version >= 3 and self.v3o["song"]:
            samp.update(self.build_song(ci, k, ann=ann))
        return samp

    def window_info(self):
        """per validation window (same order as self.items): dict(ann bool = the song has an LLM annotation, milk bool = versionid >= 10)"""
        self._load()
        L = self.L
        return dict(ann=np.array([L.ann[ci] is not None for ci, _ in self.items], bool),
                    milk=np.array([int(L.charts[ci].get("versionid", -1)) >= 10 for ci, _ in self.items], bool))

    def feat_batch(self, idx):
        """collated feature fields (fsl, fi0, fw, fvalid) for validation items `idx` (built on demand: 2000 windows of fp16
        features would be several GB).  Returns {} when feature fusion is off."""
        if not self.feats:
            return {}
        self._load()
        out = []
        for i in idx:
            ci, k = self.items[int(i)]
            L = self.L
            out.append(self._feat_window(L.sid[ci], window_times(L.tm[ci], L.first[ci], BAR * k)))
        return collate_feats(out)


def collate_feats(batch):
    """list of dict(fsl [n,D] fp16, fi0 [F], fw [F], fvalid [F]) -> zero-padded batch dict (torch tensors if torch is available)"""
    B = len(batch)
    N = max(b["fsl"].shape[0] for b in batch)
    D = batch[0]["fsl"].shape[1]
    fsl = np.zeros((B, N, D), np.float16)
    for i, b in enumerate(batch):
        fsl[i, :b["fsl"].shape[0]] = b["fsl"]
    out = {"fsl": fsl, "fi0": np.stack([b["fi0"] for b in batch]).astype(np.int32),
           "fw": np.stack([b["fw"] for b in batch]).astype(np.float32), "fvalid": np.stack([b["fvalid"] for b in batch])}
    if torch is not None:
        out = {k: torch.from_numpy(v) for k, v in out.items()}
    return out


def collate(batch, pad_id=0):
    L = max(len(b["tokens"]) for b in batch)
    L = (L + 7) // 8 * 8
    B = len(batch)
    tok = np.full((B, L), pad_id, np.int64)
    ct = np.zeros((B, L), np.int64)
    lm = np.zeros((B, L), bool)
    mel = np.empty((B, N_FRAMES, N_MEL), np.float16)
    for i, b in enumerate(batch):
        n = len(b["tokens"])
        tok[i, :n] = b["tokens"]
        ct[i, :n] = b["cur_time"]
        lm[i, :n] = b["loss_mask"]
        mel[i] = b["mel"]
    out = {"tokens": tok, "cur_time": ct, "loss_mask": lm, "mel": mel}
    if "is_fan" in batch[0]:
        out["is_fan"] = np.array([bool(b["is_fan"]) for b in batch], bool)
    if torch is not None:
        out = {k: torch.from_numpy(v) for k, v in out.items()}
    if "fsl" in batch[0]:
        out.update(collate_feats(batch))
    if "sfeat" in batch[0]:
        out.update(collate_song(batch))
    return out


def collate_song(batch):
    """list of samples with the song fields (mai.data._Source.build_song) -> zero-padded batch dict:
    sfeat f16 [B,S,1218], sscal f32 [B,S,4], sidx i64 [B,S], srel i64 [B,S], smask bool [B,S], saux f32 [B,S,5], snb / sg / sk i64 [B]"""
    B = len(batch)
    S = max(len(b["sfeat"]) for b in batch)
    sfeat = np.zeros((B, S, V3.BAR_COLS), np.float16)
    sscal = np.zeros((B, S, 4), np.float32)
    sidx = np.zeros((B, S), np.int64)
    srel = np.full((B, S), V3.REL_HALF, np.int64)
    smask = np.zeros((B, S), bool)
    saux = np.zeros((B, S, V3.AUX_COLS), np.float32)
    for i, b in enumerate(batch):
        n = len(b["sfeat"])
        sfeat[i, :n], sscal[i, :n], sidx[i, :n], srel[i, :n] = b["sfeat"], b["sscal"], b["sidx"], b["srel"]
        smask[i, :n], saux[i, :n] = b["smask"], b["saux"]
    out = dict(sfeat=sfeat, sscal=sscal, sidx=sidx, srel=srel, smask=smask, saux=saux,
               snb=np.asarray([b["snb"] for b in batch], np.int64), sg=np.asarray([b["sg"] for b in batch], np.int64),
               sk=np.asarray([b["sk"] for b in batch], np.int64))
    if "alab" in batch[0]:
        S_ = sfeat.shape[1]
        alab = np.zeros((B, S_), np.int64); apos = np.zeros((B, S_), np.int64); aend = np.zeros((B, S_), np.int64)
        aflg = np.zeros((B, S_, batch[0]["aflg"].shape[1]), np.float32)
        for i, b in enumerate(batch):
            n = len(b["alab"])
            alab[i, :n], apos[i, :n], aend[i, :n], aflg[i, :n] = b["alab"], b["apos"], b["aend"], b["aflg"]
        out.update(alab=alab, apos=apos, aend=aend, aflg=aflg)
    if torch is not None:
        out = {k: torch.from_numpy(v) for k, v in out.items()}
    return out
