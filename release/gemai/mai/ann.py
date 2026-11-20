"""seq_version 4 helpers (numpy only): LLM song-structure annotations -> per-bar features + window tokens.

Source: raw model answers `work/llm_annot/gemini-4-argon-high/sid_XXXXX.r1.json` (JSON text inside; fields used: sections[start_sec, label],
moments[time_sec, type]; the energy / mood / lead / bpm fields are NOT used, see docs/status/llm_pilot.md).

Pipeline (tools/prep_ann.py, once per chart because tempo maps differ between the slots of 7% of the songs):
  parse (strict JSON, regex fallback for the ~1.5% answers with broken quoting) -> validate -> fold the free label into the 10-word vocabulary
  -> section start times -> bar positions (chart tempo map, 4-beat bars) -> snap to the bar line with the highest MERT novelty within +-0.75 bar
  (round to the nearest bar line when no bar table is available) -> `AnnRecord` (starts, labels, moment bars).

Per-bar model inputs (`bar_arrays`; 0 = UNK = no annotation / channel off):
  lab  1..10   section label                        pos 1..10  bars since the section start (bucketed)    end 1..5  bars left in the section (bucketed)
  flg  [nb,5]  boundary (a section starts here, bar > 0) | last bar of a section (not the last section) | big_hit | riser | drop (moment at this bar line)
Window tokens (`window_tokens`): section label of the window's first bar (SEC_*), "a new section starts inside bars k+1..k+3" (SECN_*).
Training-time noise (`noisy`): section label replaced by a random one (p_label), boundary shifted by one bar (p_jit), moment dropped (p_mom).
"""
from __future__ import annotations
import json
import math
import re
import numpy as np

LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "break", "build-up", "drop", "outro", "other")
N_SEC = len(LABELS)
LABEL_ID = {n: i for i, n in enumerate(LABELS)}
_SYN = {"prechorus": "pre-chorus", "pre chorus": "pre-chorus", "pre_chorus": "pre-chorus", "buildup": "build-up", "build up": "build-up",
        "build_up": "build-up", "build": "build-up", "riser": "build-up", "post-chorus": "chorus", "post chorus": "chorus",
        "refrain": "chorus", "hook": "chorus", "final chorus": "chorus", "interlude": "break", "instrumental": "break",
        "breakdown": "break", "solo": "bridge", "middle 8": "bridge", "middle eight": "bridge", "coda": "outro", "ending": "outro",
        "opening": "intro", "introduction": "intro"}
MOMENT_KINDS = ("big_hit", "riser", "drop")          # -> flag columns 2, 3, 4
N_FLG = 2 + len(MOMENT_KINDS)
POS_EDGES = (1, 2, 3, 4, 6, 8, 12, 16, 24)           # bars since the section start -> bucket 0..9 (+1)
N_POS = len(POS_EDGES) + 1
END_EDGES = (1, 2, 4, 8)                             # bars left after this one -> bucket 0..4 (+1)
N_END = len(END_EDGES) + 1
WIN_BARS = 4
SNAP_BARS = 0.75
NOV_L = 4                                            # half-width (bars) of the checkerboard kernel used for snapping


def norm_label(s):
    """free label text -> id 0..9 (unknown words -> 'other')"""
    t = str(s).lower().strip().replace("_", "-") if s is not None else "other"
    t = _SYN.get(t, _SYN.get(t.replace("-", " "), t))
    return LABEL_ID.get(t, LABEL_ID["other"])


# ----------------------------------------------------------------------------------------------
# parsing
_NUM = r"(\"?-?[0-9]+(?::[0-9]+)*(?:\.[0-9]+)?\"?)"
_RE_SEC = re.compile(r"\"start_sec\"\s*:\s*" + _NUM + r"[^{}]*?\"end_sec\"\s*:\s*" + _NUM + r"[^{}]*?\"label\"\s*:\s*\"([^\"]*)\"", re.S)
_RE_MOM = re.compile(r"\"time_sec\"\s*:\s*" + _NUM + r"[^{}]*?\"type\"\s*:\s*\"([^\"]*)\"", re.S)


def to_sec(x):
    """number | '12.5' | '1:05.3' | '1:02:03' -> seconds, None when unusable"""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, str):
        t = x.strip().strip('"')
        try:
            v = 0.0
            for p in t.split(":"):
                v = v * 60.0 + float(p)
            return v
        except Exception:
            return None
    return None


def parse_raw(content):
    """raw model text -> (secs [(start_s, end_s, label_text)], moments [(t, type_text)], mode) ; mode in 'json' | 'regex' | None (nothing usable)"""
    t = (content or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(t[i:j + 1])
            secs, moms = [], []
            for s in d.get("sections", []) or []:
                a, b = to_sec(s.get("start_sec")), to_sec(s.get("end_sec"))
                if a is not None:
                    secs.append((a, b, str(s.get("label", "other"))))
            for mm in d.get("moments", []) or []:
                tt = to_sec(mm.get("time_sec"))
                if tt is not None:
                    moms.append((tt, str(mm.get("type", "other")).lower().strip()))
            return secs, moms, "json"
        except Exception:
            pass
    secs = [(to_sec(a), to_sec(b), l) for a, b, l in _RE_SEC.findall(t)]
    secs = [s for s in secs if s[0] is not None]
    moms = [(to_sec(a), ty.lower().strip()) for a, ty in _RE_MOM.findall(t)]
    moms = [x for x in moms if x[0] is not None]
    return (secs, moms, "regex") if secs else (None, None, None)


# ----------------------------------------------------------------------------------------------
# bar grid
def bar_novelty(Bn, L=NOV_L):
    """Bn: centred + L2-normalised bar vectors [nb, D] (mai.v3.bar_embed, zero rows = bars past the audio).  Checkerboard novelty at every
    bar line b (between bar b-1 and b): within-block similarity minus cross-block similarity of the L bars before / after.  -> float [nb+1],
    0 where either side has fewer than 2 bars.  Normalised to max 1."""
    nb = len(Bn)
    S = Bn @ Bn.T
    nov = np.zeros(nb + 1)
    for b in range(2, nb - 1):
        a0, a1 = max(0, b - L), b
        b0, b1 = b, min(nb, b + L)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        nov[b] = S[a0:a1, a0:a1].mean() + S[b0:b1, b0:b1].mean() - 2.0 * S[a0:a1, b0:b1].mean()
    nov = np.maximum(nov, 0.0)
    return nov / max(nov.max(), 1e-9)


def snap_bar(x, nb, nov=None):
    """fractional bar position x (bars from the chart origin) -> int bar line 1..nb-1 (section start).  With a novelty curve: the candidate
    within +-SNAP_BARS with the highest novelty (ties / flat curve: nearest); without: nearest."""
    lo, hi = max(1, int(math.ceil(x - SNAP_BARS))), min(nb - 1, int(math.floor(x + SNAP_BARS)))
    if hi < lo or nov is None:
        return int(min(max(round(x), 1), max(nb - 1, 1)))
    ks = np.arange(lo, hi + 1)
    sc = nov[ks] - 0.1 * np.abs(ks - x)
    return int(ks[int(np.argmax(sc))])


class AnnRecord:
    """snapped annotation of one chart: section start bars (starts[0] == 0), label ids, moment bars / kinds (MOMENT_KINDS index)"""
    __slots__ = ("nb", "starts", "labels", "mbar", "mkind")

    def __init__(self, nb, starts, labels, mbar=(), mkind=()):
        self.nb = int(nb)
        self.starts = np.asarray(starts, np.int32)
        self.labels = np.asarray(labels, np.int32)
        self.mbar = np.asarray(mbar, np.int32)
        self.mkind = np.asarray(mkind, np.int32)

    def to_dict(self):
        return dict(nb=self.nb, starts=self.starts.astype(np.int16), labels=self.labels.astype(np.int8),
                    mbar=self.mbar.astype(np.int16), mkind=self.mkind.astype(np.int8))

    @classmethod
    def from_dict(cls, d):
        return cls(d["nb"], d["starts"], d["labels"], d["mbar"], d["mkind"])


def build_record(secs, moms, tempo, first, nb, duration, Bn=None):
    """parsed answer -> (AnnRecord | None, info dict).  secs [(start_s, end_s, label_text)], moms [(t, type)], tempo: NpTempo of the chart,
    first: audio offset of beat 0, nb: bars of the chart, duration: audio length (s), Bn: bar embeddings for snapping (optional).
    Reasons for None: 'few_sections' (< 2 usable sections), 'bad_span' (the sections do not span the audio: last start/end outside 0.75..1.25 x duration)."""
    info = dict(n_raw=len(secs))
    secs = sorted(((a, b, l) for a, b, l in secs if a < duration - 1.0), key=lambda s: s[0])
    ends = [s[1] for s in secs if s[1] is not None]
    if len(secs) < 2:
        return None, dict(info, reason="few_sections")
    last_end = max(ends) if ends else secs[-1][0]
    if not (0.75 * duration <= last_end <= 1.25 * duration):
        return None, dict(info, reason="bad_span", last_end=last_end, duration=duration)
    nov = bar_novelty(Bn) if (Bn is not None and len(Bn) >= 8) else None
    starts, labels, shifts = [0], [norm_label(secs[0][2])], []
    for a, _, lab in secs[1:]:
        x = float(tempo.to_beat(max(a - first, 0.0))) / 4.0
        if x >= nb - 0.5:
            continue
        b = snap_bar(x, nb, nov)
        shifts.append(abs(b - x))
        if b <= starts[-1]:                       # the previous section collapsed to zero length: it vanishes
            if len(starts) > 1:
                starts[-1], labels[-1] = b, norm_label(lab)
            else:
                labels[-1] = norm_label(lab)
            continue
        starts.append(b)
        labels.append(norm_label(lab))
    if len(starts) < 2:
        return None, dict(info, reason="few_sections")
    mb, mk = [], []
    for t, ty in moms:
        if ty in MOMENT_KINDS:
            x = float(tempo.to_beat(max(t - first, 0.0))) / 4.0
            if x < nb:
                bar = int(min(max(round(x), 0), nb - 1))
                if (bar, MOMENT_KINDS.index(ty)) not in set(zip(mb, mk)):
                    mb.append(bar)
                    mk.append(MOMENT_KINDS.index(ty))
    info.update(n_sec=len(starts), snap_shift=float(np.mean(shifts)) if shifts else 0.0, snapped=nov is not None, n_mom=len(mb))
    return AnnRecord(nb, starts, labels, mb, mk), info


# ----------------------------------------------------------------------------------------------
# per-bar features
def noisy(rec, rng, p_label=0.10, p_jit=0.10, p_mom=0.10):
    """training-time corruption of a record (returns a new AnnRecord): section labels replaced by a random label (p_label), interior boundaries
    shifted by +-1 bar (p_jit; kept strictly increasing), moments dropped (p_mom)"""
    starts, labels = rec.starts.copy(), rec.labels.copy()
    n = len(starts)
    if p_jit > 0:
        for i in range(1, n):
            if rng.random() < p_jit:
                s = starts[i] + (1 if rng.random() < 0.5 else -1)
                if starts[i - 1] < s and (i == n - 1 or s < starts[i + 1]) and s < rec.nb:
                    starts[i] = s
    if p_label > 0:
        flip = rng.random(n) < p_label
        labels[flip] = rng.integers(0, N_SEC, int(flip.sum()))
    mb, mk = rec.mbar, rec.mkind
    if p_mom > 0 and len(mb):
        keep = rng.random(len(mb)) >= p_mom
        mb, mk = mb[keep], mk[keep]
    return AnnRecord(rec.nb, starts, labels, mb, mk)


def bar_arrays(rec):
    """AnnRecord -> dict(lab int64 [nb] 1..10, pos int64 [nb] 1..10, end int64 [nb] 1..5, flg float32 [nb, N_FLG])"""
    nb = rec.nb
    b = np.arange(nb)
    si = np.clip(np.searchsorted(rec.starts, b, side="right") - 1, 0, len(rec.starts) - 1)
    ends = np.concatenate([rec.starts[1:], [nb]])
    lab = rec.labels[si] + 1
    pos = np.searchsorted(np.asarray(POS_EDGES), b - rec.starts[si], side="right") + 1
    left = ends[si] - 1 - b
    end = np.searchsorted(np.asarray(END_EDGES), left, side="right") + 1
    flg = np.zeros((nb, N_FLG), np.float32)
    inner = rec.starts[1:]
    flg[inner[inner < nb], 0] = 1.0
    flg[(inner - 1)[(inner - 1 >= 0) & (inner - 1 < nb)], 1] = 1.0
    for bar, k in zip(rec.mbar, rec.mkind):
        if 0 <= bar < nb:
            flg[bar, 2 + int(k)] = 1.0
    return dict(lab=lab.astype(np.int64), pos=pos.astype(np.int64), end=end.astype(np.int64), flg=flg)


def unk_arrays(nb):
    return dict(lab=np.zeros(nb, np.int64), pos=np.zeros(nb, np.int64), end=np.zeros(nb, np.int64), flg=np.zeros((nb, N_FLG), np.float32))


def pool_arrays(a, g):
    """per-bar arrays -> per memory token (groups of g consecutive bars, same grouping as mai.v3.pool_table): label / position of the centre
    bar, flags = max over the group"""
    if g <= 1:
        return a
    nb = len(a["lab"])
    S = -(-nb // g)
    c = np.minimum(np.arange(S) * g + (g - 1) // 2, nb - 1)
    pad = S * g - nb
    flg = np.concatenate([a["flg"], np.zeros((pad, a["flg"].shape[1]), np.float32)]).reshape(S, g, -1).max(1)
    return dict(lab=a["lab"][c], pos=a["pos"][c], end=a["end"][c], flg=flg)


def window_tokens(a, k):
    """per-bar arrays + window start bar k -> (sec 0..9 | None, secn 0/1 | None): label of bar k; 1 when a new section starts in bars k+1..k+3"""
    if a is None or k < 0 or k >= len(a["lab"]) or a["lab"][k] == 0:
        return None, None
    nxt = a["flg"][k + 1:k + WIN_BARS, 0]
    return int(a["lab"][k]) - 1, int(nxt.max() > 0) if len(nxt) else 0


def to_json(rec):
    return dict(nb=rec.nb, starts=rec.starts.tolist(), labels=rec.labels.tolist(), mbar=rec.mbar.tolist(), mkind=rec.mkind.tolist())


def load_store(path):
    """work/llm_annot/v4_ann.pkl (tools/prep_ann.py) -> {(sid, slot): AnnRecord-dict}; {} when the file does not exist"""
    import os
    import pickle
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)["records"]
