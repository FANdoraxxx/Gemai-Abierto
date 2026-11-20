"""Inference-side helper for seq_version 3 models (song context / PLAN / REF).  Not used by training.

    ctx = SongContext(model, V, tab, tempo, seq_meta["v3"], device)     # once per song: song encoder pass + plan for every window
    w = ctx.window(k, ticks, offs, toks, batch=K)                       # per window starting at bar k
    # w["plan"]  PLAN bucket (int, or None); w["plan_token"] id; w["ref"] = (marker, tokens, cur_time) -> build_window_tokens(..., ref=w["ref"])
    # w["song"]  decoder memory dict -> model.init_cache(enc, max_len, song=w["song"]) / model(..., song=w["song"])

`tab` is the bar table of the song (mai.v3.compute_bar_table on the generation beat grid, or mai.v3.load_bar_table for a dataset song whose
tempo map is the official one).  `ticks/offs/toks` = the chart generated SO FAR in the packed layout of mai.sequence.pack_notes; only notes older
than the history range of the window are ever read, so the notes of the window being generated may be missing.  See docs/V3_INFER.md.
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import v3 as V3


class SongContext:
    def __init__(self, model, V, tab, tempo, stats=None, device=None, plan_mode="rank", ann=None):
        import torch
        self.torch = torch
        self.model, self.V, self.tempo = model, V, tempo
        self.stats = stats or V3.default_stats()
        self.dev = torch.device(device) if device is not None else next(model.parameters()).device
        self.tab = np.asarray(tab)
        self.nb = len(self.tab)
        self.g = V3.pool_factor(self.nb)
        self.Bn = V3.bar_embed(self.tab)                                  # retrieval embeddings (audio only)
        self.plan_mode = plan_mode
        # v4 (seq_version 4 / cfg.song_ann): `ann` = mai.ann.AnnRecord of the song on THIS bar grid (tools/prep_ann.py logic: parse the LLM answer,
        # mai.ann.build_record with the generation beat grid) or None = no annotation (UNK everywhere, exactly what the model saw in training dropout)
        from . import ann as A
        self.ann_arr = A.bar_arrays(ann) if ann is not None else None
        arr = V3.song_arrays(self.tab, tempo, 0)                          # srel is replaced per window; everything else is k-independent
        self._arr = arr
        S = len(arr["sfeat"])
        song = dict(sfeat=torch.from_numpy(arr["sfeat"])[None], sscal=torch.from_numpy(arr["sscal"])[None],
                    sidx=torch.from_numpy(arr["sidx"])[None], smask=torch.ones(1, S, dtype=torch.bool))
        if getattr(model.cfg, "song_ann", False):
            pa = A.pool_arrays(self.ann_arr if self.ann_arr is not None else A.unk_arrays(self.nb), self.g)
            song.update(alab=torch.from_numpy(pa["lab"])[None], apos=torch.from_numpy(pa["pos"])[None], aend=torch.from_numpy(pa["end"])[None],
                        aflg=torch.from_numpy(pa["flg"])[None])
        song = {k: v.to(self.dev) for k, v in song.items()}
        with torch.no_grad():
            was = model.training
            model.eval()
            self.summary = model.song_summary(song)                      # ONE pass over the whole song
            model.train(was)
        pred = self.summary["aux"][0, :, 0].float().cpu().numpy()
        self.pred_log_count = pred                                        # predicted log1p(notes per bar) per memory token
        self.plans = V3.plan_from_pred(pred, self.g, self.nb, self.stats["plan_edges"], self.stats.get("plan_q"), plan_mode)
        # --- v11 causality-experiment hook (default off): MAI_PLAN_OVERRIDE=<json {"plans":[bucket per bar k]}> replaces the model's own plan ---
        self.plans_model = np.array(self.plans, copy=True)
        self.plan_override_info = None
        _ov = os.environ.get("MAI_PLAN_OVERRIDE", "").strip()
        if _ov:
            with open(_ov, "r", encoding="utf-8") as _f:
                _pl = np.asarray(json.load(_f)["plans"], np.int64).reshape(-1)
            _n_in = int(len(_pl))
            if _n_in < self.nb:
                _pl = np.concatenate([_pl, np.full(self.nb - _n_in, _pl[-1] if _n_in else V3.N_PLAN // 2, np.int64)])
            _pl = np.clip(_pl[:self.nb], 0, V3.N_PLAN - 1)
            self.plans = _pl
            self.plan_override_info = dict(path=_ov, n_in=_n_in, n_bars=int(self.nb),
                                           buckets_every4=[int(x) for x in _pl[::4]],
                                           model_buckets_every4=[int(x) for x in self.plans_model[::4]])
        # --- seq_version 5 BD hook (default off -> BD*_UNK): MAI_BD_OVERRIDE=<json {"bd":[bucket per bar b]}>; window k feeds bd[k .. k+3] (bars past the end: 0)
        self.bd = None
        self.bd_override_info = None
        _bd = os.environ.get("MAI_BD_OVERRIDE", "").strip()
        if _bd and V.seq_version >= 5:
            with open(_bd, "r", encoding="utf-8") as _f:
                _b = np.asarray(json.load(_f)["bd"], np.int64).reshape(-1)
            self.bd = np.clip(_b, 0, V3.N_BD_B - 1)
            self.bd_override_info = dict(path=_bd, n_in=int(len(_b)), n_bars=int(self.nb))

    def plan_bucket(self, k, shift=0):
        """PLAN bucket for the window starting at bar k (shift: +/- buckets, e.g. to make the song more or less dynamic)"""
        b = int(self.plans[min(max(int(k), 0), self.nb - 1)]) + int(shift)
        return min(max(b, 0), V3.N_PLAN - 1)

    def window(self, k, ticks, offs, toks, batch=1, plan_shift=0, use_plan=True, use_ref=True, use_song=True, use_ann=True):
        """everything a v3 model needs for the window that starts at bar k (the target is bars k .. k+3)"""
        torch, V = self.torch, self.V
        k = int(k)
        plan = self.plan_bucket(k, plan_shift) if use_plan else None
        ref = (V["REF_NONE"], [], [])
        info = None
        if use_ref:
            st = self.stats
            mk, t, c, info = V3.retrieve_ref(V, self.Bn, ticks, offs, toks, k, st["ref_edges"], st["ref_budget"], st["ref_lite"])
            ref = (mk, t, c)
        song = None
        if use_song:
            srel = torch.from_numpy(V3.rel_index(self._centres() - float(k)))
            if self.dev.type == "cuda":
                srel = srel.pin_memory()                                   # speed: async H2D copy instead of a blocking pageable one
            song = self.model.song_memory(self.summary, srel, batch=batch)
        from .sequence import plan_token
        out = dict(plan=plan, plan_token=plan_token(V, plan), ref=ref, ref_info=info, song=song)
        if V.seq_version >= 5:                            # v5: BD1..BD4 tokens (v6: the 4 BDI ids that are forced inside the target) (UNK unless a BD plan was given)
            from .sequence import bd_tokens
            bd = None
            if self.bd is not None:
                seg = self.bd[k:k + V3.WIN_BARS]
                bd = [int(x) for x in seg] + [0] * (V3.WIN_BARS - len(seg))
            if V.seq_version == 5:
                out.update(bd=bd, bd_tokens=bd_tokens(V, bd))
            else:
                from .sequence import bdi_ids
                out.update(bd=bd, bd_tokens=None, bdi=bdi_ids(V, bd))
        if V.seq_version >= 4:                            # v4: SEC / SECN condition tokens of the window (UNK without annotation)
            from . import ann as A
            from .sequence import sec_token, secn_token
            sec = secn = None
            if use_ann and self.ann_arr is not None:
                sec, secn = A.window_tokens(self.ann_arr, k)
            out.update(sec=sec, secn=secn, sec_token=sec_token(V, sec), secn_token=secn_token(V, secn))
        return out

    def _centres(self):
        if not hasattr(self, "_c"):
            S = len(self._arr["sfeat"])
            start = np.arange(S, dtype=np.float64) * self.g
            self._c = start + (np.minimum(start + self.g, self.nb) - start - 1) / 2.0
        return self._c
