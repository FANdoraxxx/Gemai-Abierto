"""Event vocabulary, mirroring, window tokenization and grammar-aware detokenization."""
from __future__ import annotations
import copy
from .simai import Note, Arm, Seg, SHAPES, SENSORS, TPB

# window geometry (beats)
W_BEATS = 16     # predicted span
C_BEATS = 16     # history context
L_BEATS = 4      # audio lookahead
T_MIN = -C_BEATS * TPB
T_MAX = W_BEATS * TPB
MAX_WAIT = 4 * TPB
MAX_DUR_B = 64

N_LV = 141           # 1.0 .. 15.0 step 0.1
N_SLOT = 5           # 2..6
N_VER = 32
N_BPM = 40           # 40..440 step 10
N_PROG = 10
N_PLAN = 9           # seq_version 3: per-window planned density (relative to the song mean), see mai/v3.py
N_REFS = 4           # seq_version 3: similarity bucket of the retrieved earlier section (REF segment)
N_SEC = 10            # seq_version 4: LLM section label of the window's first bar (mai/ann.py LABELS), + SEC_UNK
N_SECN = 2            # seq_version 4: "a new section starts inside the window" flag (SECN_0 / SECN_1), + SECN_UNK
N_BD = 12             # seq_version 5: per-bar note-count buckets of the 4 target bars (mai/v3.py BD_EDGES), + BDj_UNK per slot
N_BD_SLOTS = 4
N_DENS = 12          # seq_version 2: chart note-density buckets (notes per SECOND of the whole chart, log-spaced edges, see mai/v2.py)
FEAT_NAMES = ("TOUCH", "EXN", "FEST")
AMT_SUFFIX = ("N", "L", "M", "H")      # seq_version 2 feature amount: none / low / mid / high (tertiles)

class Vocab:
    """seq_version 1 (default): the original v1 vocabulary.  seq_version 2 appends tokens at the END (all v1 ids unchanged):
    TOUCH_{L,M,H}, EXN_{L,M,H}, FEST_{L,M,H} (amount buckets; *_N = none, *_Y = present / amount unspecified, *_UNK as before)
    and DENS_UNK, DENS_0..DENS_{N_DENS-1}."""

    def __init__(self, designers, seq_version=1, with_fut=False):
        self.designers = list(designers)
        self.seq_version = int(seq_version)
        self.with_fut = bool(with_fut)
        toks = ["PAD", "BOS", "EOS", "CTX", "TGT"]
        toks += ["LV_UNK"] + [f"LV_{i}" for i in range(N_LV)]
        toks += ["SLOT_UNK"] + [f"SLOT_{i}" for i in range(N_SLOT)]
        toks += ["TYPE_UNK", "TYPE_STD", "TYPE_DX"]
        for f in ("TOUCH", "EXN", "FEST"):
            toks += [f"{f}_UNK", f"{f}_N", f"{f}_Y"]
        toks += ["DES_UNK", "DES_OTHER"] + [f"DES_{i}" for i in range(len(self.designers))]
        toks += ["VER_UNK"] + [f"VER_{i}" for i in range(N_VER)]
        toks += [f"BPM_{i}" for i in range(N_BPM)]
        toks += [f"PROG_{i}" for i in range(N_PROG)]
        self.time0 = len(toks)
        toks += [f"T_{t}" for t in range(T_MIN, T_MAX)]
        toks += ["N_TAP", "N_HOLD", "N_SLIDE", "N_TOUCH", "N_THOLD"]
        toks += [f"BTN_{b}" for b in range(1, 9)]
        toks += [f"SEN_{s}" for s in SENSORS]
        toks += ["BRK", "EX", "FW", "HEADLESS", "SBRK", "ARM"]
        toks += [f"SH_{s}" for s in SHAPES]
        toks += [f"WAIT_{w}" for w in range(MAX_WAIT)]
        toks += [f"DB_{b}" for b in range(MAX_DUR_B)]
        toks += [f"DT_{t}" for t in range(TPB)]
        if self.seq_version >= 2:
            for f in FEAT_NAMES:
                toks += [f"{f}_{b}" for b in AMT_SUFFIX[1:]]
            toks += ["DENS_UNK"] + [f"DENS_{i}" for i in range(N_DENS)]
        if self.seq_version >= 3:       # appended after every v2 id (v1/v2 ids unchanged): PLAN condition + REF segment marker
            toks += ["PLAN_UNK"] + [f"PLAN_{i}" for i in range(N_PLAN)]
            toks += ["REF_NONE"] + [f"REF_{i}" for i in range(N_REFS)]
        if self.seq_version >= 4:       # appended after every v3 id (v1/v2/v3 ids unchanged): LLM structure conditions
            toks += ["SEC_UNK"] + [f"SEC_{i}" for i in range(N_SEC)]
            toks += ["SECN_UNK"] + [f"SECN_{i}" for i in range(N_SECN)]
        if self.seq_version == 5:       # appended after every v4 id: BD1..BD4 = note-count bucket of target bar 1..4 (one token set per slot,
            for j in range(1, N_BD_SLOTS + 1):   # so the summed condition injection can tell the slots apart)
                toks += [f"BD{j}_UNK"] + [f"BD{j}_{i}" for i in range(N_BD)]
        if self.seq_version >= 6:       # seq_version 6 (= v4 + interleaved BD): BARM = "this bar is finished" (predicted by the model), BDI_k = note-count bucket of the
            toks += ["BARM", "BDI_UNK"] + [f"BDI_{i}" for i in range(N_BD)]      # bar that starts now (forced by the caller, never predicted)
        if self.with_fut:               # v6 infill: FUT marker at the VERY END (all previous ids unchanged)
            toks += ["FUT"]
        self.toks = toks
        self.idx = {t: i for i, t in enumerate(toks)}
        self.size = len(toks)
        g = self.idx
        self.PAD, self.BOS, self.EOS, self.CTX, self.TGT = g["PAD"], g["BOS"], g["EOS"], g["CTX"], g["TGT"]
        self.FUT = g["FUT"] if self.with_fut else None
        # header = BOS + conditions + BPM + PROG + CTX
        self.n_cond = 16 if self.seq_version == 5 else 12 if self.seq_version >= 4 else (10 if self.seq_version >= 3 else (9 if self.seq_version >= 2 else 8))
        # header = BOS + conditions + BPM + PROG [+ REF marker (v3)] + CTX ; v3 additionally has the REF events between the marker and CTX
        self.header_len = 1 + self.n_cond + 3 + (1 if self.seq_version >= 3 else 0)
        self.dens_col = 9               # position of DENS_* in the sequence (v2, v3)
        self.plan_col = 10              # position of PLAN_* in the sequence (v3)
        self.sec_col = 11               # position of SEC_* (v4): section label of the window's first bar
        self.bd_col = 13                # position of BD1_* (v5); BD2..BD4 follow (positions 13..16)
        self.bd0 = self.idx["BD1_UNK"] if self.seq_version == 5 else (self.idx["BARM"] if self.seq_version >= 6 else 0)   # first BD-block token id (ids bd0 .. bd0 + n_bd - 1)
        self.n_bd_tok = N_BD_SLOTS * (N_BD + 1) if self.seq_version == 5 else (2 + N_BD if self.seq_version >= 6 else 0)
        self.barm = self.idx["BARM"] if self.seq_version >= 6 else None
        self.bdi0 = self.idx["BDI_UNK"] if self.seq_version >= 6 else None      # BDI_UNK; BDI_b = bdi0 + 1 + b
        self.secn_col = 12              # position of SECN_* (v4): new section starts inside the window

    def __getitem__(self, k):
        return self.idx[k]

    def time_tok(self, t):
        return self.time0 + (t - T_MIN)

    def tok_time(self, i):
        return i - self.time0 + T_MIN

    def is_time(self, i):
        return self.time0 <= i < self.time0 + (T_MAX - T_MIN)

    def to_dict(self):
        d = {"designers": self.designers}
        if self.seq_version != 1:
            d["seq_version"] = self.seq_version
        if self.with_fut:
            d["with_fut"] = True
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["designers"], seq_version=d.get("seq_version", 1), with_fut=d.get("with_fut", False))

# ----------------------------------------------------------------- mirroring
H_BTN = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
V_BTN = {1: 4, 2: 3, 3: 2, 4: 1, 5: 8, 6: 7, 7: 6, 8: 5}
SWAP_SHAPE = {"CW": "CCW", "CCW": "CW", "P": "Q", "Q": "P", "PP": "QQ", "QQ": "PP", "S": "Z", "Z": "S"}

def _mir_btn(b, mode):
    if mode in ("H", "HV"): b = H_BTN[b]
    if mode in ("V", "HV"): b = V_BTN[b]
    return b

def _mir_sensor(si, mode):
    name = SENSORS[si]
    if name == "C":
        return si
    z, k = name[0], int(name[1])
    if z in "AB":
        k = _mir_btn(k, mode)
    else:
        if mode in ("H", "HV"):
            k = k if k in (1, 5) else 10 - k
        if mode in ("V", "HV"):
            k = ((6 - k) + 8) % 8 or 8
    return SENSORS.index(f"{z}{k}")

def mirror_notes(notes, mode):
    if mode in (None, "", "N"):
        return notes
    out = []
    for n in notes:
        m = copy.deepcopy(n)
        if n.kind in ("touch", "thold"):
            m.pos = _mir_sensor(n.pos, mode)
        else:
            m.pos = _mir_btn(n.pos, mode)
        for a in m.arms:
            for s in a.segs:
                s.end = _mir_btn(s.end, mode)
                if s.via: s.via = _mir_btn(s.via, mode)
                if mode in ("H", "V"):
                    s.shape = SWAP_SHAPE.get(s.shape, s.shape)
        out.append(m)
    return out

# ----------------------------------------------------------------- chart features
def chart_flags(notes):
    touch = any(n.kind in ("touch", "thold") for n in notes)
    ex = any(n.ex for n in notes)
    fest = any(len(a.segs) > 1 for n in notes if n.kind == "slide" for a in n.arms)
    return touch, ex, fest

def note_sort_key(n):
    cat = 1 if n.kind in ("touch", "thold") else 0
    return (int(n.beat), cat, n.pos, ["tap", "hold", "slide", "touch", "thold"].index(n.kind))

def canonicalize(notes):
    """sort, dedupe same-position notes at the same tick, sort slide arms"""
    notes = sorted(notes, key=note_sort_key)
    out = []; seen = set()
    for n in notes:
        cat = 1 if n.kind in ("touch", "thold") else 0
        k = (int(n.beat), cat, n.pos)
        if k in seen:
            continue
        seen.add(k)
        if n.kind == "slide":
            n.arms = sorted(n.arms, key=lambda a: (a.wait, [(s.shape, s.end, s.via) for s in a.segs]))
        out.append(n)
    return out

# ----------------------------------------------------------------- encoding
def _dur_toks(V, d):
    d = max(1, int(d))
    b, t = divmod(d, TPB)
    b = min(b, MAX_DUR_B - 1)
    return [V[f"DB_{b}"], V[f"DT_{t}"]]

def encode_note(V, n):
    t = []
    if n.kind in ("touch", "thold"):
        t.append(V["N_TOUCH" if n.kind == "touch" else "N_THOLD"])
        t.append(V[f"SEN_{SENSORS[n.pos]}"])
        if n.fw: t.append(V["FW"])
        if n.kind == "thold": t += _dur_toks(V, n.dur)
        return t
    t.append(V[{"tap": "N_TAP", "hold": "N_HOLD", "slide": "N_SLIDE"}[n.kind]])
    t.append(V[f"BTN_{n.pos}"])
    if n.brk: t.append(V["BRK"])
    if n.ex: t.append(V["EX"])
    if n.kind == "hold":
        t += _dur_toks(V, n.dur)
    elif n.kind == "slide":
        if n.headless: t.append(V["HEADLESS"])
        for a in n.arms:
            t.append(V["ARM"])
            if int(a.wait) != TPB:
                t.append(V[f"WAIT_{min(MAX_WAIT - 1, max(0, int(a.wait)))}"])
            k = 0
            for ns, d in a.groups:
                for s in a.segs[k:k + ns]:
                    t.append(V[f"SH_{s.shape}"])
                    if s.shape == "GRANDV": t.append(V[f"BTN_{s.via}"])
                    t.append(V[f"BTN_{s.end}"])
                k += ns
                t += _dur_toks(V, d)
            if a.brk: t.append(V["SBRK"])
    return t

def encode_events(V, notes, t0, tmin, tmax):
    """notes: canonical tick notes. Emit tokens for notes with t0+tmin <= beat < t0+tmax,
    times relative to t0."""
    out = []; last = None
    for n in notes:
        rt = int(n.beat) - t0
        if rt < tmin or rt >= tmax:
            continue
        if rt != last:
            out.append(V.time_tok(rt)); last = rt
        out += encode_note(V, n)
    return out

# ----------------------------------------------------------------- decoding (tolerant)
def decode_events(V, toks, t0=0):
    """parse event tokens (no specials) back into tick notes; times += t0. Malformed notes are dropped."""
    notes = []; cur_t = None
    i = 0; n_tok = len(toks)
    T = V.toks
    def name(k):
        return T[toks[k]] if k < n_tok else "EOS"
    def read_dur(k):
        a, b = name(k), name(k + 1)
        if a.startswith("DB_") and b.startswith("DT_"):
            return int(a[3:]) * TPB + int(b[3:]), k + 2
        return None, k
    while i < n_tok:
        nm = name(i)
        if V.is_time(toks[i]):
            cur_t = V.tok_time(toks[i]) + t0; i += 1; continue
        if cur_t is None or not nm.startswith("N_"):
            i += 1; continue
        kind = nm[2:].lower(); i += 1
        try:
            if kind in ("touch", "thold"):
                s = name(i)
                if not s.startswith("SEN_"): continue
                n = Note("touch" if kind == "touch" else "thold", cur_t, SENSORS.index(s[4:])); i += 1
                if name(i) == "FW": n.fw = True; i += 1
                if kind == "thold":
                    d, i2 = read_dur(i)
                    if d is None: continue
                    n.dur = d; i = i2
                notes.append(n); continue
            b = name(i)
            if not b.startswith("BTN_"): continue
            n = Note(kind, cur_t, int(b[4:])); i += 1
            if name(i) == "BRK": n.brk = True; i += 1
            if name(i) == "EX": n.ex = True; i += 1
            if kind == "hold":
                d, i2 = read_dur(i)
                if d is None: continue
                n.dur = d; i = i2
            elif kind == "slide":
                if name(i) == "HEADLESS": n.headless = True; i += 1
                ok = True
                while name(i) == "ARM":
                    i += 1
                    wait = TPB
                    if name(i).startswith("WAIT_"): wait = int(name(i)[5:]); i += 1
                    segs, groups, pending = [], [], 0
                    while name(i).startswith("SH_"):
                        sh = name(i)[3:]; i += 1; via = 0
                        if sh == "GRANDV":
                            if not name(i).startswith("BTN_"): ok = False; break
                            via = int(name(i)[4:]); i += 1
                        if not name(i).startswith("BTN_"): ok = False; break
                        segs.append(Seg(sh, int(name(i)[4:]), via)); i += 1; pending += 1
                        d, i2 = read_dur(i)
                        if d is not None:
                            groups.append((pending, d)); pending = 0; i = i2
                    if not ok or not segs or not groups:
                        ok = False; break
                    if pending:
                        groups[-1] = (groups[-1][0] + pending, groups[-1][1])
                    a = Arm(wait, segs, groups)
                    if name(i) == "SBRK": a.brk = True; i += 1
                    n.arms.append(a)
                if not ok or not n.arms: continue
            elif kind != "tap":
                continue
            notes.append(n)
        except (ValueError, IndexError):
            continue
    return notes
