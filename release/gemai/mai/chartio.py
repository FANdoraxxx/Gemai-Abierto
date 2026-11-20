"""Helpers to get tick notes + tempo for the analysis tools (playcheck / chartstats / levelest / compare)."""
from __future__ import annotations
import os, json, pickle
import numpy as np
from .tokens import Vocab, decode_events, canonicalize
from .tempo import NpTempo
from .simai import load_song, quantize, TPB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V = None

def work(*p):
    return os.path.join(ROOT, "work", *p)

def vocab():
    global _V
    if _V is None:
        meta = json.load(open(work("meta.json"), encoding="utf-8"))
        _V = Vocab([d["name"] for d in meta["designers"]])
    return _V

def load_charts(path=None):
    """list of chart dicts from work/charts.pkl (metadata + tokenized notes)"""
    with open(path or work("charts.pkl"), "rb") as stream:
        return pickle.load(stream)

def chart_notes(c, mode="N"):
    """decode tokenized chart dict -> canonical tick notes (list of mai.simai.Note)"""
    V = vocab()
    m = c["modes"][mode]
    t0 = V.time_tok(0)
    toks, offs, ticks = m["toks"], m["offs"], m["ticks"]
    out = []
    for i in range(len(ticks)):
        out += decode_events(V, [t0] + toks[offs[i]:offs[i + 1]].tolist(), int(ticks[i]))
    return out

def chart_tempo(c):
    """Tempo map with the chart's absolute audio origin in ``tempo.first``."""
    tempo = NpTempo(c["beats"], c["bpms"])
    tempo.first = float(c.get("first", 0.0))
    return tempo

def official_notes(folder, slot, root=None):
    """reparse maidata.txt -> (canonical tick notes, NpTempo, meta)"""
    root = root or os.path.join(ROOT, "official charts")
    meta, songs = load_song(os.path.join(root, folder))
    notes, tm = songs[slot]
    q, _ = quantize(notes)
    tempo = NpTempo(tm.beats, tm.bpms)
    tempo.first = float(tm.first)
    return canonicalize(q), tempo, meta

def dir_chart(root, folder, slot=None):
    """chart from <root>/<folder>/maidata.txt (e.g. generated output). slot=None -> the highest slot present.
    returns (canonical tick notes, NpTempo with .first, slot) or None; empty inotes are preserved."""
    path = os.path.join(root, folder)
    if not os.path.exists(os.path.join(path, "maidata.txt")):
        return None
    meta, songs = load_song(path)
    if not songs:
        return None
    if slot is None:
        slot = max(songs)
    if slot not in songs:
        return None
    notes, tm = songs[slot]
    q, _ = quantize(notes)
    tempo = NpTempo(tm.beats, tm.bpms)
    tempo.first = float(tm.first)
    return canonicalize(q), tempo, slot
