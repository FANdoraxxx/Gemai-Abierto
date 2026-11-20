"""Beat-aligned, grammar- and hand-constrained chart generation."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import activity, audio, beatgrid, calib as calib_mod, chorus as chorus_mod, feats as feats_mod, handtarget, handtravel, levelest, levelpred, murigate, playcheck, qmode, rerank, simai, steer as steer_mod
from . import v3 as V3, v3_infer
from .grammar import EventGrammar
from .ffmpeg_util import ffmpeg_exe
from .model import ChartModel, DecodeCache, ModelConfig
from .sequence import (TGT, bpm_progress_tokens, build_window_tokens, condition_tokens,
                       pack_notes, unk_condition_row, window_mel, bdi_ids)
from .tempo import NpTempo
from types import SimpleNamespace
from .steer import FeatureBias, Steer
from . import plan as plan_mod
from .tokens import Vocab, TPB, T_MAX, T_MIN, N_DENS, canonicalize, chart_flags, decode_events, encode_events

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GenerationOptions:
    level: str = "auto"
    auto_level: bool = True          # --level auto: predict the level from the audio (mai/levelpred.py) and condition on it
    auto_level_spread: float = 1.3   # stretch of the predicted level around the training mean (the ridge shrinks towards it)
    auto_level_bias: float = 0.0     # added to the predicted level
    slot: str = "master"
    touch: str = "auto"              # v2 on -> MID amount; off -> N plus grammar mask; auto -> UNK
    ex: str = "auto"
    fest: str = "auto"
    fest_seg_gate: bool = True        # stop connected slides at 2 segments (0 = raw v6 behaviour)
    fest_seg_max: int = 3             # v6d2_seg3: the gate penalises only arms that already have >= this many segments (v6d2_plus behaviour = 2)
    designer: str = ""
    version: str = "auto"
    chart_type: str = "auto"
    bpm: float | None = None
    bpm_map: str = ""                # explicit piecewise tempo map "beat:bpm,beat:bpm" (beats are quarter notes from beat 0)
    tempo_changes: str = "auto"      # auto/on: infer local tempos; off: reproduce the constant sp0 grid
    first: float | None = None
    grid_from: str | None = None
    temperature: float = 0.9         # timing tokens (time ticks, EOS)
    top_p: float = 0.95              # nucleus truncation of the timing tokens only
    content_temperature: float = 1.0  # note kinds, lanes, slide shapes, modifiers, durations
    content_top_p: float = 1.0       # no truncation: rare shapes and kinds must stay reachable
    seed: int = 0
    cfg: float = 1.0                 # classifier-free guidance weight against the all-UNK prefix
    cfg_level: float = 2.0           # CFG on LV (v2 also clears its derived DENS in the reference); needs --level
    cfg_touch: float = 1.0           # CFG on the TOUCH token (needs --touch on); reference is cfg_flag_ref
    cfg_fest: float = 1.0            # CFG on the FEST token (needs --fest on)
    cfg_bd: float = 1.0              # seq_version 5: CFG on the BD1..BD4 per-bar density tokens (reference row = BD*_UNK); > 1 needs a BD plan (MAI_BD_OVERRIDE)
    cfg_flag_ref: str = "n"          # unconditional reference for the flag CFGs: n (flag N) or unk
    touch_strength: float = 1.0      # --touch on: logit bias scale on N_TOUCH / N_THOLD at note-kind positions
    touch_target: float = 0.065       # --touch on: touch share of all notes the feedback aims at (official touch charts: mean .086, median .074)
    fest_rate: float = 3.0           # --fest on: connected slides per minute the feedback aims at (official FESTiVAL charts: median 2.7)
    fest_strength: float = 1.0       # --fest on: logit bias scale on continuing a slide arm with another SH_ segment
    candidates: int = 1              # whole-chart candidates (best by playability, level error, likelihood)
    window_candidates: int = 6       # per-window candidates K, reranked by the penalties below (1 = off); round 2: 4, 6 gives the muri gate room
    w_logprob: float = 0.0           # weight of the mean token log-probability in the window score (>0 favours typical, repetitive windows)
    w_activity: float = 0.5          # audio-activity vs note mismatch penalty weight
    w_repeat: float = 1.0            # repeated-bar penalty weight
    repeat_pair: float = 0.5         # ... cost of a bar identical to the previous one (3+ in a row always costs 2+)
    repeat_far: float = 0.0          # ... cost of a bar identical to an earlier non-adjacent bar (motif repetition)
    w_density: float = 1.0           # density-vs-level penalty weight (only with --level)
    w_intensity: float = 0.0         # weight of the penalty for windows whose notes per half-bar do not follow the audio flux (rank correlation; 0 = off)
    density_steer: float = 1.0       # strength of the notes-per-window feedback on the gap tilt (needs --level; 0 = off)
    density_scale: float = 1.0       # multiplier on the note count expected for the requested level
    act_prior: float = 0.0           # gain of a feed-forward density tilt ~ log(expected density at the gap start / song mean): notes follow the audio intensity (0 = off; needs a level)
    density_scale_hi: float = 0.23    # extra density multiplier per level constant above 13.5 (requested 14 / 14+ came out 0.2-0.5 low)
    style_steer: float = 0.5         # strength of the online share feedback (hold / slide / break / ex / doubles)
    mod_gain: float = 1.0            # gain of the BRK / EX / touch share feedback (style_steer scales it too)
    mod_memory: int = 3              # windows of history of the BRK / EX / touch share measurement (0 = the whole chart)
    kind_prior: float = 0.0          # gain of the audio-conditional kind prior (mai/audiobars.py)
    style_memory: int = 3            # windows of history of the hold / slide / double share feedback (0 = the whole chart)
    mod_target_scale: float = 1.0    # multiplier on the official BRK / EX / touch shares the feedback aims at
    touch_steer: float = 0.0         # share feedback on touch notes while --touch auto (strength; 0 = off)
    rhythm_steer: float = 0.5        # share feedback on 12th / 16th / 24th / 32nd onset ticks (strength; 0 = off)
    rhythm_target_scale: float = 1.0  # multiplier on the official rhythm shares
    rhythm_classes: str = "16th,24th"  # rhythm classes the feedback acts on
    brk_bias: float = -0.7            # constant logit offset on BRK (added to the calibration and the feedback)
    brk_timing_gate: bool = False      # musical break gate (opt-in): phrase/beat placement, per-bar cooldown, tap-chord symmetry
    brk_audio_floor: float = 0.50      # final gate: drop BRK notes below this whole-song onset-flux quantile (0 = off)
    ex_auto: float = -0.6             # constant logit offset on EX while --ex auto (the unknown-flag context over-produces EX)
    touch_auto: float = -0.25          # constant logit offset on touch notes while --touch auto
    cont_steer: float = 0.0          # share feedback on connected slide arms while --fest auto (strength; 0 = off)
    cont_target_scale: float = 1.0   # multiplier on the official connected-arm share (2.9 %)
    shape_steer: float = 0.0         # share feedback on the slide-segment shapes (STRAIGHT / CW / CCW / VEE / P / Q / PP / QQ / GRANDV) towards the official MiLK+ shares of the level (strength; 0 = off)
    shape_prior: float = 30.0        # pseudo-segments at the target share in the running share estimate
    shape_gain: float = 0.4          # integrator gain of the shape knobs (times shape_steer)
    shape_targets: str = ""          # shape_targets.json (default: <package>/work/shape_targets.json)
    motif: float = 3.0               # logit boost for continuing a bar that equals the start of an earlier bar (motif repetition)
    motif_min: int = 1               # leading notes that must match before the boost starts
    motif_end: float = 0.5           # relative penalty on extra notes once the reference bar is complete
    motif_lookback: int = 32         # bars searched for a reference
    anti_repeat: float = 1.5         # logit penalty for extending a bar pattern that already repeated (0 = off)
    calib: bool = True               # v1 default: work/calibration_v1.json; v2 requires an explicit compatible file
    calib_file: str = ""             # v2 files must declare meta.seq_version=2; refit for the checkpoint family
    calib_skip: str = ""             # comma list of calibration groups to leave out (shape,cont,kind,touch,thold,double,brk,sbrk,ex,gap)
    max_target_tokens: int = 900
    max_windows: int = 0
    max_segments: int = 4            # segments per slide arm (official charts: 99.8% have <= 4)
    muri: bool = True                # window-level muri gate in the reranker (mai/murigate.py): reject / penalise muri events
    muri_resamples: int = 2          # extra decode rounds of a window when every candidate has a hard muri event
    muri_w_hard: float = 5.0         # reranker penalty per hard muri event (Overlap counts double)
    muri_w_soft: float = 1.0         # scale of the soft penalties (inner screen, non-touch multi-touch; tail / outer from muri_hard_level)
    muri_resample_penalty: float = 1.0  # also resample a window when its best candidate's muri penalty is at least this (0 = hard events only)
    muri_slack: float = 0.1          # events the chart may carry beyond the official per-level budget at no full penalty (see murigate.allowance)
    muri_hard_level: float = 14.0    # tail collisions and outer-key hits are hard events below this requested / predicted level
    muri_check: bool = True          # whole-chart MiaCode check of the final chart, recorded in gen_info.json
    slide_mask: bool = True          # grammar: only slide segments that exist in the game (mai/slidetable.py); False = round-2 behaviour
    max_backtracks: int = 96         # per attempt; exceeding it resamples the window, then keeps a partial window
    window_seconds: float = 90.0     # wall-clock safety net per window (all candidates together)
    max_resamples: int = 2           # fresh-seed retries of a stuck candidate before the partial fallback
    prune_end: bool = True           # reject tokens that cannot finish before the end of the audio (test hook)
    feats: str = "auto"              # feature-fusion checkpoints: auto (precomputed / cached / extract), off (mel-only fallback) or a .npy path
    v3_song: bool = True             # seq_version 3: song-level memory (song encoder over all bars); False = exact trained fallback (no song path)
    v3_ref: bool = True              # seq_version 3: REF segment from the notes generated so far; False = REF_NONE
    v3_plan: bool = True             # seq_version 3: PLAN token predicted from the audio; False = PLAN_UNK
    plan_shift: int = 0              # seq_version 3: +/- buckets added to every predicted PLAN (more / less dynamic)
    plan_mode: str = "rank"          # seq_version 3: rank (predicted window ratios rank-matched to the training distribution) or raw (fixed edges)
    plan_density: float = 0.0        # seq_version 3: expected note count of the density terms x (planned ratio of the window) ** this (0 = off)
    expect_mode: str = "level"       # expected notes per window: level = official nps of the level at this tempo x the window's relative intensity (audio + PLAN), table = the old flux table
    expect_hi: float = 0.0           # expect_mode level: extra expected-density multiplier per level constant above 13.5
    density_band: float = 0.35       # expect_mode level: dead band (log) of the two-sided density penalty
    guard: bool = True               # expect_mode level: candidate guard (empty playing half-bars, floor, streak) ranks before the score
    guard_empty: int = 3             # ... empty half-bars (of 8, while the audio plays) that make a candidate a violation (0 = off)
    guard_floor: float = 0.4         # ... a playing window below this share of its expected notes is a violation (0 = off)
    guard_streak: float = 0.6        # ... two playing windows in a row below this share are a violation (0 = off)
    guard_play_db: float = 3.0       # ... a half-bar plays when its loudness is within this many dB of the song median
    guard_rounds: int = 1            # extra decode rounds (denser tilt) when every candidate of a window violates the guard
    guard_shift: float = 0.4         # ... tilt shift of the first extra round (the second one uses 1.5x)
    # ---- quality mode (mai/qmode.py): spend inference time instead of retraining; -1 = the preset of ``quality`` decides
    quality: str = "fast"            # fast (current) | fine (K=16 + lookahead + 1 refine pass) | max (fine + 2 passes + best of 2 charts)
    la_top: int = -1                 # lookahead: the top-M candidates get a sampled continuation (0 = off)
    la_samples: int = -1             # ... samples of the next window per candidate
    la_weight: float = 0.5           # weight of the continuation value in the commit score
    la_lp: float = 2.0               # weight of the continuation's mean log-prob inside that value
    w_hand: float = -1.0             # weight of the hand term (mai/handtravel.py / handtarget.py) in window scores (0 = off); -1 = preset
    hand_mode: str = "target"        # target: match the official hand-movement distribution of the level (mai/handtarget.py); travel: penalise travel (fine v1)
    hand_targets: str = ""           # hand_targets.json (default: <package>/work/hand_targets.json)
    hand_kappa: float = 1.0          # target mode: scale of the budget penalty
    hand_z0: float = 0.25            # target mode: dead band of the standardized cumulative deviation (units of the chart-to-chart spread)
    hand_n0: float = 120.0           # target mode: pseudo-contacts distributed like the target at the start of a chart
    hand_hard: float = 1.0           # target mode: weight of the implausible-event terms
    hand_cross_w: float = 2.0        # target mode: weight of the crossing-rate bins relative to the speed / jump-size bins
    qguard: bool = True               # quality mode: density / activity guard (a window may not be much sparser / denser than the level + audio expect, nor leave active music empty)
    qguard_rounds: int = 2            # quality mode: extra decode rounds (shifted density tilt) when every candidate of a window violates the guard
    qguard_resample: float = 0.15     # ... triggered when the least violating clean candidate exceeds this guard violation
    qguard_w: float = 3.0             # cost per unit of guard violation in the lookahead value, the refine badness and the refine candidate cost
    refine_keep: float = 0.7         # refine never accepts a replacement with fewer notes than this share of the original (unless it still holds >= 90 % of the expected notes)
    refine_passes: int = -1          # self-refine passes after the whole chart (0 = off)
    refine_frac: float = 0.15        # share of windows regenerated per pass (worst first)
    refine_k: int = 0                # candidates per regenerated window (0 = window_candidates)
    refine_eval: int = 6             # candidates that get the expensive terms (muri region, teacher-forced log-prob)
    refine_after: int = 1            # existing windows after the regenerated one that enter the rule / hand / muri cost
    refine_lp: float = 0.02          # cost per nat of log-prob of the affected following windows
    refine_margin: float = 0.1       # a replacement must be cheaper than the existing window by this
    refine_min_bad: float = 0.25     # windows with a lower cost are never regenerated
    refine_lpz: float = 0.15         # cost per std of mean log-prob below the song average (target selection)
    refine_max_affected: int = 3     # following windows whose log-prob enters a candidate's cost
    refine_tail_bars: int = 8        # skip self-refine targets starting in the final N bars of the audio (0 = off)
    # ---- v6 infill refinement (FUT token): odd / even / worst passes with the infill model (mai/infill.py)
    infill_passes: int = 0           # 0 = off (byte-identical to the build without infill); 3 = odd, even, worst non-adjacent
    infill_frac: float = 0.25        # share of windows regenerated in the odd / even passes
    infill_k: int = 16               # candidates per regenerated window
    infill_min_bad: float = 0.0      # never regenerate a window below this 5-part cost
    infill_margin: float = 0.10      # a replacement must reduce the regional cost by at least this
    infill_checkpoint: str = ""      # path of the infill checkpoint (recorded in gen_info; the loader attaches the model)
    infill_source: str = ""          # optional existing chart directory: infill-refine this draft instead of regenerating pass 1
    infill_seed_offset: int = 0      # fresh candidate bank for R2/R3 when refining the same source charts
    infill_round: int = 0            # additive evolution-round salt for the infill RNG (in addition to infill_seed_offset)
    # ---- seq_version 4 song-structure annotation channel (mai/ann.py): section labels / boundaries fed to the song encoder + SEC / SECN tokens
    ann: str = "none"                # none (default: every bar and window token is UNK, exactly the training dropout case) |
                                     # file:<path> (v4 annotation JSON: record {nb,starts,labels,mbar,mkind} or the raw LLM answer
                                     # {sections:[{start_sec,end_sec,label}],moments:[{time_sec,type}]}) |
                                     # auto (reserved: whole-song plan from the audio; the product format is not settled yet -> behaves as none) |
                                     # pred (sections predicted from the audio by scratch/plan/sectionpred.py)
    # ---- whole-song planning layer (mai/plan.py, scratch/plan): default 0 = off = byte-identical to the build without it
    plan: float = 0.0                # strength S of the plan (0 = off; 1 = the planner's full chorus / verse contrast; every planned quantity enters as value ** S)
    plan_file: str = ""              # planner JSON of this song (default: predict the sections from the audio and plan online)
    plan_parts: str = "dens,token,tilt,rhythm"   # dens token tilt rhythm wide share ff creat
    plan_ff: float = 1.0             # gain of the feed-forward offsets: logit = gain x log(planned share multiplier); density tilt = gain x (E_window - E_mean) / 16 beats
    # ---- chorus / highlight helper (mai/chorus.py): default 0 = off = byte-identical to the build without it
    chorus: float = 0.0              # strength S of the chorus parts (0 = off; 1 = the official chorus / verse contrast at full strength)
    chorus_parts: str = "dens,pool,pick"          # dens pool stream pick
    chorus_k: float = 1.0            # pool: extra window candidates of a chorus window, x window_candidates (1 = double)
    chorus_fine_ff: float = 0.6      # pool: logit offset on the non-8th grid time tokens of a chorus window
    chorus_fine_mode: str = "any"    # pool: "any" = every non-8th tick, "grid" = only the 16th / 24th / 32nd grid
    chorus_stream_ff: float = 0.5    # stream: logit offset on the time token that continues a running 16th stream
    chorus_pick: float = 1.5         # pick: weight of the rerank penalty for a chorus window far from the official chorus medians
    chorus_pick_band: float = 0.10   # pick: dead band (log) of that penalty
    chorus_band: float = 0.0         # dens: density dead band inside the chorus windows (0 = keep --density-band)
    chorus_min_frac: float = 0.5     # a window is a chorus window when at least this share of its bars is chorus / drop
    chorus_targets: str = ""         # targets JSON (default: out/plan/targets.json of the project)
    chorus_section: str = "pred"     # pred (audio-only section predictor) | file:<json> (stored record {nb,starts,labels})
    chorus_onset_q: float = 0.7      # rwo: a fine tick is re-weighted only when its onset strength reaches this quantile of the song's tick strengths
    chorus_onset_tol: float = 1.0    # rwo: +- ticks around a tick position searched for an onset peak (mai.chorus.onset_strength)

    def validate(self):
        parse_level(self.level)
        if self.infill_passes < 0 or self.infill_k < 1 or not (0.0 < self.infill_frac <= 1.0):
            raise ValueError("infill_passes must be >= 0, infill_k >= 1 and 0 < infill_frac <= 1")
        if not math.isfinite(self.auto_level_spread) or self.auto_level_spread <= 0 or not math.isfinite(self.auto_level_bias):
            raise ValueError("auto_level_spread must be positive and auto_level_bias finite")
        if self.slot not in ("master", "remaster"):
            raise ValueError("slot must be master or remaster")
        for name in ("touch", "ex", "fest"):
            if getattr(self, name) not in ("auto", "on", "off"):
                raise ValueError(f"{name} must be auto, on or off")
        if self.chart_type not in ("auto", "std", "dx"):
            raise ValueError("chart_type must be auto, std or dx")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not math.isfinite(self.cfg) or self.cfg < 1:
            raise ValueError("cfg must be finite and >= 1")
        if self.bpm is not None and (not math.isfinite(self.bpm) or self.bpm <= 0):
            raise ValueError("bpm must be finite and positive")
        if self.tempo_changes not in ("auto", "on", "off"):
            raise ValueError("tempo_changes must be auto, on or off")
        if self.bpm_map:
            parse_bpm_map(self.bpm_map)
            if self.bpm is not None:
                raise ValueError("bpm and bpm_map are mutually exclusive")
            if self.grid_from:
                raise ValueError("grid_from and bpm_map are mutually exclusive")
        if self.first is not None and not math.isfinite(self.first):
            raise ValueError("first must be finite")
        if self.candidates < 1 or self.max_windows < 0 or self.max_target_tokens < 8:
            raise ValueError("candidates >= 1, max_windows >= 0 and max_target_tokens >= 8 required")
        if self.window_candidates < 1 or self.window_candidates > 16:
            raise ValueError("window_candidates must be in [1, 16]")
        for name in ("cfg_level", "cfg_touch", "cfg_fest", "cfg_bd"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be finite and >= 1")
        if not math.isfinite(self.content_temperature) or self.content_temperature < 0:
            raise ValueError("content_temperature must be finite and nonnegative")
        if not math.isfinite(self.content_top_p) or not 0 < self.content_top_p <= 1:
            raise ValueError("content_top_p must be in (0, 1]")
        for name in ("touch_strength", "fest_strength", "w_logprob", "w_activity", "w_repeat", "w_density", "w_intensity",
                     "density_steer", "density_scale", "act_prior", "style_steer", "anti_repeat"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.cfg_flag_ref not in ("n", "unk"):
            raise ValueError("cfg_flag_ref must be n or unk")
        if not (self.ann in ("none", "auto", "pred") or self.ann.startswith("file:")):
            raise ValueError("ann must be none, auto, pred or file:<path>")
        if not math.isfinite(self.plan) or self.plan < 0 or not math.isfinite(self.plan_ff) or self.plan_ff < 0:
            raise ValueError("plan and plan_ff must be finite and >= 0")
        plan_mod.parse_parts(self.plan_parts)
        for name in ("muri_w_hard", "muri_w_soft", "muri_hard_level", "muri_resample_penalty", "muri_slack"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.muri_resamples < 0 or self.muri_resamples > 8:
            raise ValueError("muri_resamples must be in [0, 8]")
        if self.expect_mode not in ("level", "table"):
            raise ValueError("expect_mode must be level or table")
        for name in ("shape_steer", "shape_prior", "shape_gain"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        for name in ("expect_hi", "density_band", "guard_floor", "guard_streak", "guard_play_db", "guard_shift"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.guard_empty < 0 or self.guard_rounds < 0 or self.guard_rounds > 4:
            raise ValueError("guard_empty >= 0 and guard_rounds in [0, 4] required")
        if self.plan_mode not in ("rank", "raw"):
            raise ValueError("plan_mode must be rank or raw")
        if not isinstance(self.plan_shift, int) or abs(self.plan_shift) > 8:
            raise ValueError("plan_shift must be an integer in [-8, 8]")
        if not math.isfinite(self.plan_density) or self.plan_density < 0:
            raise ValueError("plan_density must be finite and >= 0")
        if not math.isfinite(self.chorus) or self.chorus < 0:
            raise ValueError("chorus must be finite and >= 0")
        for name in ("chorus_k", "chorus_fine_ff", "chorus_stream_ff", "chorus_pick", "chorus_pick_band", "chorus_min_frac",
                     "chorus_band", "chorus_onset_tol"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.chorus_min_frac > 1:
            raise ValueError("chorus_min_frac must be in [0, 1]")
        if not math.isfinite(self.chorus_onset_q) or not 0.0 <= self.chorus_onset_q <= 1.0:
            raise ValueError("chorus_onset_q must be in [0, 1]")
        if self.chorus_fine_mode not in ("any", "grid"):
            raise ValueError("chorus_fine_mode must be any or grid")
        if self.chorus > 0:
            chorus_mod.parse_parts(self.chorus_parts)
            if not (self.chorus_section in ("pred", "") or str(self.chorus_section).startswith("file:")):
                raise ValueError("chorus_section must be pred or file:<path>")
        if self.quality not in ("fast", "fine", "max"):
            raise ValueError("quality must be fast, fine or max")
        if self.hand_mode not in ("target", "travel"):
            raise ValueError("hand_mode must be target or travel")
        for name in ("hand_kappa", "hand_z0", "hand_n0", "hand_hard", "hand_cross_w", "qguard_w", "qguard_resample", "refine_keep"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        for name in ("la_weight", "la_lp", "refine_frac", "refine_lp", "refine_margin", "refine_min_bad", "refine_lpz"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.la_samples > 4 or self.la_top > 8 or self.refine_passes > 4 or not 0 <= self.refine_frac <= 1:
            raise ValueError("la_samples <= 4, la_top <= 8, refine_passes <= 4, refine_frac in [0, 1] required")
        if self.max_segments < 1 or self.max_backtracks < 1 or self.max_resamples < 0 or self.window_seconds <= 0:
            raise ValueError("max_segments >= 1, max_backtracks >= 1, max_resamples >= 0, window_seconds > 0 required")


def parse_level(text):
    """Display integers use x.2, '+' uses x.7; decimal values are exact."""
    text = str(text).strip().lower()
    if text == "auto":
        return None
    if re.fullmatch(r"\d+\+", text):
        value = int(text[:-1]) + 0.7
    elif re.fullmatch(r"\d+", text):
        value = int(text) + 0.2
    else:
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"invalid level {text!r}; use auto, 13, 13+ or 13.7") from exc
    if not math.isfinite(value) or not 1 <= value <= 15:
        raise ValueError("level constant must be between 1.0 and 15.0")
    return round(value, 1)


DENSITY_KNEE = 13.5             # density_scale_hi acts on the level constant above this
GUARD_MURI_PEN = 3.0            # a guard-clean candidate whose muri penalty is at least this does not count as an acceptable pick


def _muri_pen(entry):
    parts = entry[5]
    return float(parts.get("p_muri", 0.0) or 0.0)


def _guard_tier(entry):
    """Ranking tier of a (clean) candidate entry: guard violations first, a heavy muri penalty counts as one more violation."""
    parts = entry[5]
    return int(parts.get("guard", 0)) + (1 if _muri_pen(entry) >= GUARD_MURI_PEN else 0)
AUTO_LEVEL_MEAN = 13.4           # train-split mean MASTER level constant (fallback and centre of the spread)


def _effective_bpm(grid):
    """Duration-weighted main BPM of a tempo map (identical to the single-BPM path when there is one tempo)."""
    bpms = np.asarray(grid["bpms"], np.float64)
    if len(bpms) <= 1:
        return float(bpms[0]) if len(bpms) else 150.0
    beats = np.asarray(grid["beats"], np.float64)
    first = float(grid.get("first", 0.0))
    duration = float(grid.get("debug", {}).get("duration", 0.0) or 0.0)
    ends = first + np.concatenate([
        [0.0], np.cumsum(np.diff(beats) * 60.0 / bpms[:-1])])
    if duration > first:
        ends = np.append(ends, duration)
    else:
        ends = np.append(ends, ends[-1] + 60.0 / bpms[-1])
    weights = {}
    for i, value in enumerate(bpms):
        seconds = max(0.0, min(ends[i + 1], duration or ends[i + 1])
                      - max(ends[i], first))
        weights[float(value)] = weights.get(float(value), 0.0) + seconds
    return max(weights, key=weights.get)


def prune_breaks_by_audio(notes, mel, tempo, first, floor=0.5):
    """Remove BRK modifiers whose onset is not a local audio-flux peak (gate-off path untouched)."""
    if floor <= 0.0 or mel is None or tempo is None:
        return notes
    flux = activity.flux_frames(mel)
    if len(flux) < 2:
        return notes
    order = np.sort(flux)
    for n in notes:
        brk = bool(getattr(n, "brk", False)) or any(bool(getattr(a, "brk", False)) for a in (getattr(n, "arms", None) or ()))
        if not brk:
            continue
        sec = float(getattr(tempo, "first", first) or first) + float(tempo.to_sec(int(round(float(n.beat))) / TPB))
        f = int(np.clip(round(sec * activity.FPS), 0, len(flux) - 1))
        vals = flux[max(0, f - 2):min(len(flux), f + 3)]
        v = float(np.max(vals)) if len(vals) else float(flux[f])
        rank = float(np.searchsorted(order, v, side="right")) / max(1, len(order))
        if rank < float(floor):
            n.brk = False
            for a in (getattr(n, "arms", None) or ()):
                a.brk = False
    return notes


def _window_seconds(grid, start_tick, end_tick):
    """Real seconds spanned by the ticks [start_tick, end_tick) of a (possibly variable-BPM) grid."""
    tempo = NpTempo(grid["beats"], grid["bpms"])
    return float(tempo.to_sec(float(end_tick) / TPB) - tempo.to_sec(float(start_tick) / TPB))


def auto_level_from_audio(mel, grid, options, log=print):
    """Level constant to condition on when the user gives none: ridge prediction from the audio (see mai/levelpred.py)."""
    bpm = _effective_bpm(grid)
    try:
        predictor = levelpred.get_predictor()
        raw = predictor.predict(mel, bpm)
        source = "work/levelpred.json"
    except OSError:
        log("warning: work/levelpred.json missing; auto level falls back to the training mean", flush=True)
        raw, source = AUTO_LEVEL_MEAN, "fallback"
    centre = AUTO_LEVEL_MEAN
    level = centre + options.auto_level_spread * (raw - centre) + options.auto_level_bias
    level = round(float(min(max(level, 12.0), 14.7)), 1)
    return dict(level=level, raw=round(float(raw), 3), bpm=bpm, source=source)


def resolve_designers(text, names):
    if not text:
        return [(-2, "UNK")]
    result = []
    for query in text.split(","):
        query = query.strip()
        if not query:
            raise ValueError("empty designer name")
        if query.upper() in ("OTHER", "UNK"):
            hit = (-1 if query.upper() == "OTHER" else -2, query.upper())
        else:
            exact = [(i, n) for i, n in enumerate(names) if n.casefold() == query.casefold()]
            matches = exact or [(i, n) for i, n in enumerate(names) if query.casefold() in n.casefold()]
            if len(matches) != 1:
                reason = "ambiguous" if matches else "unknown"
                raise ValueError(f"{reason} designer {query!r}; valid: " + ", ".join(names))
            hit = matches[0]
        if hit not in result:
            result.append(hit)
    return result


def resolve_version(text, official_root=ROOT / "official charts"):
    text = str(text).strip()
    if text.casefold() == "auto":
        return -1
    if text.isdigit():
        value = int(text)
        if 0 <= value < 32:
            return value
        raise ValueError("version id must be in [0, 31]")
    versions = {}
    for path in Path(official_root).glob("*/maidata.txt"):
        header = {}
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                if line.startswith("&inote_"):
                    break
                if line.startswith("&version"):
                    key, _, value = line[1:].strip().partition("=")
                    header[key] = value
        if header.get("versionid", "").isdigit() and header.get("version"):
            versions[header["version"].casefold()] = int(header["versionid"])
    exact = [value for name, value in versions.items() if name == text.casefold()]
    hits = exact or list({value for name, value in versions.items() if text.casefold() in name})
    if len(hits) != 1:
        raise ValueError(f"unknown or ambiguous version {text!r}; valid: " + ", ".join(sorted(versions)))
    return hits[0]


def guidance_plan(options):
    """Per-designer guidance rows in order: [(name, weight, token position, reference token name)]."""
    plan = []
    if parse_level(options.level) is not None and options.cfg_level > 1:
        plan.append(("level", options.cfg_level, 0, "LV_UNK"))
    ref = "N" if options.cfg_flag_ref == "n" else "UNK"
    if options.touch == "on" and options.cfg_touch > 1:
        plan.append(("touch", options.cfg_touch, 3, f"TOUCH_{ref}"))
    if options.fest == "on" and options.cfg_fest > 1:
        plan.append(("fest", options.cfg_fest, 5, f"FEST_{ref}"))
    if getattr(options, "cfg_bd", 1.0) > 1:
        plan.append(("bd", options.cfg_bd, 12, "BD1_UNK"))        # last block; rows_with_window keeps its BD slots UNK
    return plan


def n_condition_rows(options, n_designers):
    return n_designers * (1 + len(guidance_plan(options))) + int(options.cfg > 1)


def designer_rows(options, n_rows):
    """Number of designer (conditional) rows among ``n_rows`` prefix rows."""
    n = (n_rows - int(options.cfg > 1)) // (1 + len(guidance_plan(options)))
    if n < 1 or n_condition_rows(options, n) != n_rows:
        raise ValueError("prefix row count does not match the guidance layout")
    return n


def _validate_v2_density_meta(meta):
    """Require the checkpoint's fitted density mapping; never substitute another run's work/v2_stats.json."""
    message = ("seq_version=2 checkpoint requires fitted seq_meta.dens_edges and "
               "seq_meta.level_bucket (levels and median_dens); use a complete v2 checkpoint")
    try:
        edges = np.asarray(meta["dens_edges"], np.float64)
        levels = np.asarray(meta["level_bucket"]["levels"], np.float64)
        densities = np.asarray(meta["level_bucket"]["median_dens"], np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if (edges.ndim != 1 or edges.size != N_DENS - 1 or not np.isfinite(edges).all()
            or (edges <= 0).any() or (np.diff(edges) <= 0).any()
            or levels.ndim != 1 or not levels.size or densities.shape != levels.shape
            or not np.isfinite(levels).all() or (np.diff(levels) <= 0).any()
            or not np.isfinite(densities).all() or (densities <= 0).any()):
        raise ValueError(message)
    return meta


def make_conditions(V, options, designers, version, seq_meta=None):
    """Prefix condition rows: designer rows, then per-designer guidance rows, then the all-UNK row (cfg > 1)."""
    slot = 5 if options.slot == "master" else 6
    level = parse_level(options.level)
    if V.seq_version >= 2 and level is not None:
        _validate_v2_density_meta(seq_meta)
    rows = []
    for des_id, _ in designers:
        if V.seq_version >= 2:
            from .v2 import v2_condition_row
            # MID is the explicit-on default for all three v2 amount conditions.
            amount = lambda state: "mid" if state == "on" else state
            row = v2_condition_row(
                V, seq_meta, level=level, slot=slot, chart_type=options.chart_type,
                touch=amount(options.touch), ex=amount(options.ex), fest=amount(options.fest),
                des_id=des_id, versionid=version, dens="auto")
        else:
            chart = dict(lv=level, slot=slot, dx=options.chart_type == "dx",
                         touch=options.touch == "on", ex=options.ex == "on",
                         fest=options.fest == "on", des_id=des_id, versionid=version)
            row = condition_tokens(V, chart)
            if options.chart_type == "auto":
                row[2] = V["TYPE_UNK"]
            for i, (name, stem) in enumerate((("touch", "TOUCH"), ("ex", "EXN"), ("fest", "FEST")), 3):
                if getattr(options, name) == "auto":
                    row[i] = V[f"{stem}_UNK"]
        rows.append(row)
    base = list(rows)
    for name, _, position, reference in guidance_plan(options):
        for row in base:
            other = row.copy()
            if name != "bd":                       # bd: the base row already carries BD*_UNK (set per window in rows_with_window)
                other[position] = V[reference]
            if V.seq_version >= 2 and name == "level":
                # DENS is derived from LV: the level reference must forget both.
                # Feature references retain the requested density and all other amounts.
                other[8] = V["DENS_UNK"]
            rows.append(other)
    if options.cfg > 1:
        rows.append(unk_condition_row(V))
    return rows


def rows_with_plan(V, options, rows, token):
    """seq_version 3: the condition rows of one window with the PLAN slot set (the all-UNK classifier-free row keeps PLAN_UNK)."""
    return rows_with_window(V, options, rows, token)


def rows_with_window(V, options, rows, token, sec_token=None, secn_token=None, bd_tokens=None):
    """The condition rows of one window with the per-window slots set: PLAN (seq_version 3) and SEC / SECN (seq_version 4).
    The all-UNK classifier-free row keeps the UNK tokens of every slot."""
    if V.seq_version < 3:
        return rows
    out = []
    bd_ref = range(0)                               # rows of the BD guidance block (reference = BD*_UNK): keep UNK
    if V.seq_version == 5 and getattr(options, "cfg_bd", 1.0) > 1:
        gp = guidance_plan(options)
        nd = (len(rows) - int(options.cfg > 1)) // (1 + len(gp))
        bd_ref = range(nd * len(gp), nd * (1 + len(gp)))        # "bd" is the last guidance block
    for i, row in enumerate(rows):
        row = row.copy()
        if not (options.cfg > 1 and i == len(rows) - 1):
            row[V.plan_col - 1] = token
            if V.seq_version >= 4:
                row[V.sec_col - 1] = V["SEC_UNK"] if sec_token is None else sec_token
                row[V.secn_col - 1] = V["SECN_UNK"] if secn_token is None else secn_token
            if V.seq_version == 5:
                row[V.bd_col - 1:V.bd_col + 3] = [V[f"BD{j}_UNK"] for j in range(1, 5)] if (bd_tokens is None or i in bd_ref) else bd_tokens
        out.append(row)
    return out


def load_generation_calibration(V, options, *, log=print):
    """Keep the v1 default unchanged; v2 calibration is opt-in and explicitly tagged for v2."""
    if not options.calib:
        return None
    if V.seq_version >= 2 and not options.calib_file:
        log(f"warning: seq_version={V.seq_version} checkpoint: default work/calibration_v1.json disabled; "
            f"pass --calib-file with a refitted file declaring meta.seq_version={V.seq_version}", flush=True)
        return None
    try:
        bias = calib_mod.load_bias(options.calib_file or None)
    except OSError:
        log("warning: calibration file missing; marginal calibration disabled", flush=True)
        return None
    if V.seq_version >= 2:
        meta = bias.get("meta")
        if not isinstance(meta, dict) or meta.get("seq_version") not in (V.seq_version, 4 if V.seq_version >= 5 else V.seq_version):   # v5 = v4 + BD tokens: same marginals file
            raise ValueError(f"calibration file {options.calib_file!r} is incompatible with seq_version=2; "
                             "refit for this v2 checkpoint family and declare meta.seq_version=2")
    elif bias.get("meta", {}).get("seq_version", 1) != 1:
        raise ValueError("v1 generation cannot use a calibration tagged for another sequence version")
    return calib_mod.Calibration(V, bias, use_cont=options.fest == "on",
                                 skip=[g for g in options.calib_skip.split(",") if g])


def stable_checkpoint(path):
    """Read a private snapshot, retrying if a trainer replaces the source."""
    path = Path(path).resolve()
    error = None
    for _ in range(3):
        try:
            before = path.stat()
            with tempfile.TemporaryDirectory(prefix="maiai-weights-") as folder:
                snapshot = Path(folder) / "model.pt"
                shutil.copyfile(path, snapshot)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise OSError("checkpoint changed during snapshot")
                digest = hashlib.sha256()
                with snapshot.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
            return checkpoint, dict(path=str(path), size=before.st_size, mtime_ns=before.st_mtime_ns,
                                    sha256=digest.hexdigest(), step=int(checkpoint.get("step", 0)))
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            error = exc
    raise RuntimeError(f"cannot load stable checkpoint {path}: {error}") from error


def parse_bpm_map(text):
    """Parse ``beat:bpm,beat:bpm`` into a validated TempoMap-style pair."""
    if not text:
        return [0.0], []
    beats, bpms = [], []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            raise ValueError("empty bpm_map entry")
        if ":" not in item:
            raise ValueError("bpm_map entries must be beat:bpm")
        beat_s, bpm_s = item.split(":", 1)
        try:
            beat, bpm = float(beat_s), float(bpm_s)
        except ValueError as exc:
            raise ValueError("bpm_map entries must be numbers") from exc
        if not math.isfinite(beat) or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError("bpm_map beats must be finite and bpm must be positive")
        if beats and beat <= beats[-1]:
            raise ValueError("bpm_map beats must be strictly increasing")
        beats.append(beat)
        bpms.append(bpm)
    if not beats or abs(beats[0]) > 1e-8:
        raise ValueError("bpm_map must start at beat 0")
    return beats, bpms


def resolve_grid(audio_path, options, device):
    if options.grid_from:
        path = Path(options.grid_from)
        if path.is_dir():
            path = path / "maidata.txt"
        meta, bodies = simai.read_maidata(path)
        slot = 5 if options.slot == "master" else 6
        if not bodies:
            raise ValueError(f"no chart in {path}")
        # The body is used only to read its tempo changes. Notes never enter history.
        body = bodies[slot] if slot in bodies else bodies[max(bodies)]
        _, tempo = simai.parse_chart(body, float(meta.get("wholebpm") or 120),
                                    float(meta.get("first") or 0))
        result = dict(first=tempo.first, beats=tempo.beats, bpms=tempo.bpms,
                      debug=dict(source="maidata", path=str(path.resolve())))
    elif options.bpm_map:
        beats, bpms = parse_bpm_map(options.bpm_map)
        if options.first is None:
            # The audio tracker still supplies the first downbeat; explicit tempos come from the map.
            aligned = beatgrid.estimate_grid(
                audio_path, device=str(device), bpm_hint=bpms[0],
                params={"detect_changes": False})
            first = float(aligned["first"])
        else:
            first = float(options.first)
        result = dict(first=first, beats=beats, bpms=bpms,
                      debug=dict(source="bpm_map", map=str(options.bpm_map)))
    elif options.bpm is not None and options.first is not None:
        result = dict(first=options.first, beats=[0.0], bpms=[options.bpm],
                      debug=dict(source="manual"))
    else:
        detect = options.tempo_changes in ("auto", "on")
        result = beatgrid.estimate_grid(
            audio_path, device=str(device), bpm_hint=options.bpm,
            params={"detect_changes": bool(detect)})
        result["debug"]["source"] = "automatic"
    if options.first is not None:
        result["first"] = options.first
    b, p = np.asarray(result["beats"]), np.asarray(result["bpms"])
    if (len(b) != len(p) or not len(b) or not np.isfinite(b).all()
            or not np.isfinite(p).all() or (p <= 0).any() or abs(b[0]) > 1e-8
            or (np.diff(b) <= 0).any() or not math.isfinite(float(result["first"]))):
        raise ValueError("invalid beat grid")
    return result


class _GuardTables:
    """Token class tables for one vocabulary (built once)."""
    def __init__(self, V):
        self.boundary = [False] * V.size
        self.position = [False] * V.size       # BTN_* / SEN_*
        self.sensor = [False] * V.size
        self.dt = [False] * V.size
        for i, name in enumerate(V.toks):
            if i == V.EOS or V.is_time(i) or name.startswith("N_") or i == getattr(V, "barm", None):
                self.boundary[i] = True
            elif name.startswith("BTN_"):
                self.position[i] = True
            elif name.startswith("SEN_"):
                self.position[i] = self.sensor[i] = True
            elif name.startswith("DT_"):
                self.dt[i] = True
        self.slide = V["N_SLIDE"]
        self.arm = V["ARM"]
        self.headless = V["HEADLESS"]


_TABLES = {}


def _tables(V):
    key = id(V)
    if key not in _TABLES or _TABLES[key][0] is not V:
        _TABLES[key] = (V, _GuardTables(V))
    return _TABLES[key][1]


class HandGuard:
    """Commit complete notes only; tentative grammar branches do not mutate hands.

    Besides the two-hand constraint it prunes sustained notes that cannot finish before ``end_tick`` (the end of the
    audio) using the grammar's timing floors, so the last window never explores doomed slide / hold subtrees.
    """
    def __init__(self, V, tempo, start_tick=0, end_tick=None, tracker=None, prune_end=True):
        self.V, self.tempo, self.start_tick, self.end_tick = V, tempo, start_tick, end_tick
        self.tracker = tracker or playcheck.HandTracker(tempo)
        self.holds = []
        self.recent = []                 # committed notes (absolute ticks, in order) incl. the last bars of the history
        self.committed_count = 0
        self.prune_end = prune_end
        self._t = _tables(V)
        self._memo = {}                  # (pending note tokens, cur_time) -> feasible?  valid for the current hand state

    def clone(self):
        other = HandGuard(self.V, self.tempo, self.start_tick, self.end_tick, self.tracker.copy(), self.prune_end)
        other.holds = list(self.holds)
        other.recent = list(self.recent)
        other.committed_count = self.committed_count
        return other

    def restore(self, snapshot):
        """Reset hand state in place to a ``clone()`` taken earlier."""
        self.tracker = snapshot.tracker.copy()
        self.holds = list(snapshot.holds)
        self.recent = list(snapshot.recent)
        self.committed_count = snapshot.committed_count
        self._memo = {}

    def _note(self, grammar):
        notes = decode_events(self.V, [self.V.time_tok(grammar.cur_time)] +
                              list(grammar.pending_note_tokens or ()), self.start_tick)
        return notes[0] if len(notes) == 1 else None

    def can_note(self, note):
        if self.end_tick is not None and note.end_beat() > self.end_tick:
            return False
        if note.kind in ("tap", "hold", "slide") and not (note.kind == "slide" and note.headless):
            if any(n.pos == note.pos and n.beat <= note.beat < n.beat + n.dur for n in self.holds):
                return False
        return self.tracker.can_add(note)

    def _feasible(self, grammar):
        """can_note of the grammar's pending note, memoised for the current hand state (exact: same function, same state)."""
        key = (tuple(grammar.note_tokens), grammar.cur_time)
        hit = self._memo.get(key)
        if hit is None:
            note = self._note(grammar)
            hit = self._memo[key] = note is not None and self.can_note(note)
        return hit

    @staticmethod
    def _extended(note, extra):
        """Copy of a decoded hold / touch-hold / slide whose last duration is ``extra`` ticks longer."""
        out = copy(note)
        if note.kind == "slide":
            arms = list(note.arms)
            last = arms[-1]
            groups = list(last.groups)
            groups[-1] = (groups[-1][0], groups[-1][1] + extra)
            arms[-1] = simai.Arm(last.wait, last.segs, groups, last.brk, last.ex)
            out.arms = arms
        else:
            out.dur = note.dur + extra
        return out

    def _dt_feasible(self, grammar, tokens):
        """``[pending_ok(probe)]`` for the DT tokens ``tokens`` (increasing) of a duration_tick state, exactly, with the
        hand check shared between the 48 candidate durations (``HandTracker.can_add_variants``)."""
        base = tuple(grammar.note_tokens)
        cur = grammar.cur_time
        slow = (type(self).can_note is not HandGuard.can_note or not hasattr(self.tracker, "can_add_variants")
                or len(tokens) < 3)
        if not slow:
            probe = grammar.clone()
            probe.accept(tokens[0])
            note0 = self._note(probe)
            if note0 is None:
                oks = [False] * len(tokens)
            else:
                notes = [self._extended(note0, int(x) - int(tokens[0])) for x in tokens]
                oks = [False] * len(tokens)
                if not (note0.kind in ("tap", "hold", "slide") and not (note0.kind == "slide" and note0.headless)
                        and any(n.pos == note0.pos and n.beat <= note0.beat < n.beat + n.dur for n in self.holds)):
                    idx = [i for i, n in enumerate(notes)
                           if self.end_tick is None or n.end_beat() <= self.end_tick]
                    for i, ok in zip(idx, self.tracker.can_add_variants([notes[i] for i in idx])):
                        oks[i] = ok
            for x, ok in zip(tokens, oks):
                self._memo[(base + (int(x),), cur)] = ok
            return oks
        oks = []
        for x in tokens:
            probe = grammar.clone()
            probe.accept(x)
            oks.append(not probe.pending_note_tokens or self.pending_ok(probe))
        return oks

    def pending_ok(self, grammar):
        if grammar.completed_count < self.committed_count:
            return True  # previous note was committed before the current branch frame
        return self._feasible(grammar)

    def commit(self, grammar):
        if grammar.pending_note_tokens and grammar.completed_count >= self.committed_count:
            note = self._note(grammar)
            if note is None or not self._feasible(grammar) or not self.tracker.add(note):
                raise RuntimeError("attempted to commit a hand-infeasible note")
            if note.kind == "hold":
                self.holds.append(note)
            self.holds = [n for n in self.holds if n.beat + n.dur > note.beat]
            self.recent.append(note)
            self.committed_count = grammar.completed_count + 1
            self._memo = {}

    def allowed(self, grammar, remaining):
        """Legal next tokens as an int64 array (grammar order preserved)."""
        t = self._t
        if remaining < 12:
            tokens, lo, hi = grammar.allowed_with_budget(remaining), 0, 0
        else:
            local, tail, (lo, hi) = grammar.allowed_parts()
            tokens = local + tail
        V = self.V
        stale = grammar.completed_count < self.committed_count
        pending = bool(grammar.pending_note_tokens)
        pending_ok = None
        state, kind = grammar.state, grammar.current_kind
        limit = None
        if self.prune_end and self.end_tick is not None and kind in ("hold", "thold", "slide"):
            limit = self.end_tick - (self.start_tick + grammar.cur_time)
        slide_room = (self.end_tick is None or not self.prune_end
                      or self.start_tick + grammar.cur_time + TPB + 1 <= self.end_tick)
        if state == "duration_tick" and len(tokens) and all(t.dt[x] for x in tokens):
            if stale:
                return np.zeros(0, np.int64)
            cands = [x for x in tokens if limit is None or grammar.end_floor(x) <= limit]
            if not cands:
                return np.zeros(0, np.int64)
            oks = self._dt_feasible(grammar, cands)
            return np.asarray([x for x, ok in zip(cands, oks) if ok], np.int64)
        allowed = []
        for token in tokens:
            if t.boundary[token]:
                if pending:
                    if pending_ok is None:
                        pending_ok = self.pending_ok(grammar)
                    if not pending_ok:
                        continue
                if token == t.slide and not slide_room:
                    continue  # a slide with the default one-beat wait could not finish before the audio ends
                allowed.append(token)
                continue
            if stale:
                continue  # rollback may revisit the boundary, never extend its committed note
            if t.position[token]:
                if state == "position" and kind != "slide":
                    touch = kind in ("touch", "thold")
                    pos = simai.SENSORS.index(V.toks[token][4:]) if touch else int(V.toks[token][4:])
                    note = simai.Note("touch" if touch else "tap", self.start_tick + grammar.cur_time, pos)
                    if not self.can_note(note):
                        continue
            elif state == "head" and kind == "slide" and token == t.arm:
                bodies = grammar.note_tokens
                if t.headless not in bodies:
                    note = simai.Note("tap", self.start_tick + grammar.cur_time,
                                      int(V.toks[bodies[1]][4:]))
                    if not self.can_note(note):
                        continue
            if limit is not None and grammar.end_floor(token) > limit:
                continue
            if t.dt[token]:
                probe = grammar.clone()
                probe.accept(token)
                if probe.pending_note_tokens and not self.pending_ok(probe):
                    continue
            allowed.append(token)
        out = np.asarray(allowed, np.int64)
        if hi > lo:
            # time tokens are boundary tokens: all legal or none (hand feasibility of the pending note)
            if pending and not (self.pending_ok(grammar) if pending_ok is None else pending_ok):
                return out
            out = np.concatenate([out, grammar._v.times_arr[lo:hi]])
        return out


def timing_mask(V):
    """Boolean [V.size]: time ticks and EOS (sampled with --temperature / --top-p); all other tokens are content."""
    mask = np.zeros(V.size, bool)
    mask[V.time0:V.time0 + (T_MAX - T_MIN)] = True
    mask[V.EOS] = True
    if getattr(V, "barm", None) is not None:
        mask[V.barm] = True
    return mask


_TEMP_CACHE = {}


def _sample_token_original(logits, allowed, options, rng, n_designers, plan=None, bias=None, timing=None):
    """EXACT v6_chorus_d2_preview sampling path (content_temperature == content_top_p == 1.0)."""
    values = np.asarray(logits, np.float64)
    plan = guidance_plan(options) if plan is None else plan
    cond = values[:n_designers]
    combined = cond
    row = n_designers
    for _, weight, _, _ in plan:
        combined = combined + (weight - 1.0) * (cond - values[row:row + n_designers])
        row += n_designers
    if options.cfg > 1:
        combined = combined + (options.cfg - 1.0) * (cond - values[row][None])
    values = combined
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite model logits (try --amp none)")
    tilt = None
    if isinstance(bias, steer_mod.StepBias):
        bias, tilt = bias.dense, bias.tilt
    if bias is not None:
        values = values + bias[None, :]
    allowed = np.asarray(allowed, np.int64)
    if timing is None or options.temperature == 0:
        values = values / (options.temperature if options.temperature else 1.0)
        timing_allowed = None
    else:
        key = (id(timing), options.temperature, options.content_temperature)
        divisor = _TEMP_CACHE.get(key)
        if divisor is None:
            if len(_TEMP_CACHE) > 16:
                _TEMP_CACHE.clear()
            divisor = _TEMP_CACHE[key] = (timing, np.maximum(
                np.where(timing, options.temperature, options.content_temperature), 1e-6)[None, :])
        values = values / divisor[1]
        timing_allowed = timing[allowed]
    values -= values.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(values).sum(axis=1, keepdims=True))
    log_allowed = values[:, allowed] - log_norm              # == (values - log_norm)[:, allowed], elementwise
    anchor = log_allowed.max(axis=0)
    log_mixture = anchor + np.log(np.exp(log_allowed - anchor[None]).mean(axis=0))
    probabilities = np.exp(log_mixture - log_mixture.max())
    probabilities /= probabilities.sum()
    if tilt is not None:
        # mass-preserving: re-weight the time tokens among themselves (EOS and note tokens keep their probability)
        where = np.full(values.shape[1], -1, np.int64)
        where[allowed] = np.arange(len(allowed))
        idx = where[tilt[0]]
        ok = idx >= 0
        idx, vals = idx[ok], tilt[1][ok]
        if len(idx) > 1:
            mass = probabilities[idx].sum()
            scaled = probabilities[idx] * np.exp(vals - vals.max())
            if scaled.sum() > 0:
                probabilities[idx] = scaled * (mass / scaled.sum())
    if options.temperature == 0:
        index = int(np.argmax(probabilities))
    elif timing_allowed is None:
        order = np.argsort(-probabilities, kind="stable")
        # Retain the first token crossing p, including it even at small p.
        count = min(len(order), int(np.searchsorted(np.cumsum(probabilities[order]), options.top_p)) + 1)
        choices = order[:count]
        weights = probabilities[choices]
        index = int(rng.choice(choices, p=weights / weights.sum()))
    else:
        keep = np.ones(len(allowed), bool)
        for subset, top_p in ((timing_allowed, options.top_p), (~timing_allowed, options.content_top_p)):
            members = np.flatnonzero(subset)
            if top_p >= 1.0 or len(members) < 2:
                continue
            mass = probabilities[members]
            order = np.argsort(-mass, kind="stable")
            count = min(len(order), int(np.searchsorted(np.cumsum(mass[order]) / mass.sum(), top_p)) + 1)
            keep[members[order[count:]]] = False
        weights = np.where(keep, probabilities, 0.0)
        index = int(rng.choice(len(allowed), p=weights / weights.sum()))
    return int(allowed[index]), float(math.log(max(float(probabilities[index]), 1e-30)))


def sample_token(logits, allowed, options, rng, n_designers, plan=None, bias=None, timing=None, grammar=None):
    """Combine rows (designers, guidance, all-UNK), mix designers' probabilities, sample from the allowed set.

    ``logits`` rows: ``n_designers`` conditional rows, then ``len(plan)`` blocks of ``n_designers`` guidance rows, then
    one all-UNK row when ``options.cfg > 1``.  Guidance composes additively: ``c + sum_g (w_g - 1) * (c - u_g)``.
    ``bias`` = dense [V] logit offsets added after guidance and before temperature, or a ``steer.StepBias`` that also
    carries a mass-preserving re-weighting of the time tokens (the density tilt).  ``timing`` (``timing_mask``) splits
    the vocabulary: timing tokens use ``temperature`` / ``top_p``, all others ``content_temperature`` /
    ``content_top_p`` (without it every token uses the first pair).

    The exact v6_chorus_d2_preview path is kept untouched when both content knobs are at 1.0.  A
    content temperature / top-p other than 1.0 takes the mass-preserving after_note split (the EACH vs
    timing-step mass stays at ``temperature``) and caps the content temperature at 1.0 for the
    position / button states so lane selection cannot be heated above the trained distribution.
    """
    if options.content_temperature == 1.0 and options.content_top_p == 1.0:
        return _sample_token_original(logits, allowed, options, rng, n_designers, plan, bias, timing)
    values = np.asarray(logits, np.float64)
    plan = guidance_plan(options) if plan is None else plan
    cond = values[:n_designers]
    combined = cond
    row = n_designers
    for _, weight, _, _ in plan:
        combined = combined + (weight - 1.0) * (cond - values[row:row + n_designers])
        row += n_designers
    if options.cfg > 1:
        combined = combined + (options.cfg - 1.0) * (cond - values[row][None])
    values = combined
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite model logits (try --amp none)")
    tilt = None
    if isinstance(bias, steer_mod.StepBias):
        bias, tilt = bias.dense, bias.tilt
    if bias is not None:
        values = values + bias[None, :]
    allowed = np.asarray(allowed, np.int64)
    g_state = grammar.state if grammar is not None else None
    c_temp = float(options.content_temperature)
    if g_state in ("position", "button") and c_temp > 1.0:
        c_temp = 1.0
    c_top_p = options.content_top_p
    seg_masks = None
    seg_base_values = None
    if (g_state == "segment_end" and grammar is not None
            and c_temp != options.temperature and options.temperature > 0):
        # At segment_end the allowed set mixes DB_* (stop the slide) and SH_* (chain another
        # segment).  content_temperature must only reshape the choice *within* each group; the
        # SH-vs-DB mass itself stays at the timing temperature, so heating content can never
        # inflate the Festival-slide continuation rate.
        names = [grammar.V.toks[int(tok)] for tok in allowed]
        sh_mask = np.asarray([name.startswith("SH_") for name in names], bool)
        db_mask = np.asarray([name.startswith("DB_") for name in names], bool)
        if sh_mask.any() and db_mask.any():
            seg_masks = (sh_mask, db_mask)
            seg_base_values = values.copy()
    if timing is None or options.temperature == 0:
        values = values / (options.temperature if options.temperature else 1.0)
        timing_allowed = None
    else:
        timing_allowed = timing[allowed]
        if timing_allowed.any() and (~timing_allowed).any() and c_temp != options.temperature:
            # Mass-preserving split at after_note: keep the EACH vs timing-step mass at options.temperature
            values = values / max(float(options.temperature), 1e-6)
        else:
            key = (id(timing), options.temperature, c_temp)
            divisor = _TEMP_CACHE.get(key)
            if divisor is None:
                if len(_TEMP_CACHE) > 16:
                    _TEMP_CACHE.clear()
                divisor = _TEMP_CACHE[key] = (timing, np.maximum(
                    np.where(timing, options.temperature, c_temp), 1e-6)[None, :])
            values = values / divisor[1]
    values -= values.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(values).sum(axis=1, keepdims=True))
    log_allowed = values[:, allowed] - log_norm              # == (values - log_norm)[:, allowed], elementwise
    anchor = log_allowed.max(axis=0)
    log_mixture = anchor + np.log(np.exp(log_allowed - anchor[None]).mean(axis=0))
    probabilities = np.exp(log_mixture - log_mixture.max())
    probabilities /= probabilities.sum()
    if seg_masks is not None:
        base = seg_base_values / max(float(options.temperature), 1e-6)
        base -= base.max(axis=1, keepdims=True)
        base_log_norm = np.log(np.exp(base).sum(axis=1, keepdims=True))
        base_log_allowed = base[:, allowed] - base_log_norm
        base_anchor = base_log_allowed.max(axis=0)
        base_log_mixture = base_anchor + np.log(np.exp(base_log_allowed - base_anchor[None]).mean(axis=0))
        base_prob = np.exp(base_log_mixture - base_log_mixture.max())
        base_prob /= base_prob.sum()
    else:
        base_prob = None
    if (timing_allowed is not None and timing_allowed.any() and (~timing_allowed).any()
            and options.temperature > 0 and c_temp != options.temperature):
        c_idx = np.flatnonzero(~timing_allowed)
        if len(c_idx) >= 2:
            c_mass = float(probabilities[c_idx].sum())
            if c_mass > 0.0:
                ratio = max(float(options.temperature), 1e-6) / max(float(c_temp), 1e-6)
                lp = np.log(np.maximum(probabilities[c_idx], 1e-30)) * ratio
                w_c = np.exp(lp - lp.max())
                w_c /= w_c.sum()
                probabilities[c_idx] = w_c * c_mass
    if tilt is not None:
        # mass-preserving: re-weight the time tokens among themselves (EOS and note tokens keep their probability)
        where = np.full(values.shape[1], -1, np.int64)
        where[allowed] = np.arange(len(allowed))
        idx = where[tilt[0]]
        ok = idx >= 0
        idx, vals = idx[ok], tilt[1][ok]
        if len(idx) > 1:
            mass = probabilities[idx].sum()
            scaled = probabilities[idx] * np.exp(vals - vals.max())
            if scaled.sum() > 0:
                probabilities[idx] = scaled * (mass / scaled.sum())
    if seg_masks is not None:
        for mask in seg_masks:
            mass = float(base_prob[mask].sum())
            group = probabilities[mask]
            total = float(group.sum())
            if total > 0.0:
                probabilities[mask] = group * (mass / total)
    if options.temperature == 0:
        index = int(np.argmax(probabilities))
    elif timing_allowed is None:
        order = np.argsort(-probabilities, kind="stable")
        # Retain the first token crossing p, including it even at small p.
        count = min(len(order), int(np.searchsorted(np.cumsum(probabilities[order]), options.top_p)) + 1)
        choices = order[:count]
        weights = probabilities[choices]
        index = int(rng.choice(choices, p=weights / weights.sum()))
    else:
        keep = np.ones(len(allowed), bool)
        groups = (((seg_masks[0], c_top_p), (seg_masks[1], c_top_p)) if seg_masks is not None
                  else ((timing_allowed, options.top_p), (~timing_allowed, c_top_p)))
        for subset, top_p in groups:
            members = np.flatnonzero(subset)
            if top_p >= 1.0 or len(members) < 2:
                continue
            mass = probabilities[members]
            order = np.argsort(-mass, kind="stable")
            count = min(len(order), int(np.searchsorted(np.cumsum(mass[order]) / mass.sum(), top_p)) + 1)
            keep[members[order[count:]]] = False
        weights = np.where(keep, probabilities, 0.0)
        index = int(rng.choice(len(allowed), p=weights / weights.sum()))
    return int(allowed[index]), float(math.log(max(float(probabilities[index]), 1e-30)))


@torch.no_grad()
def rows_step(model, tok, cur_time, cache, pos, last=None):
    """One decoder step for a batch whose rows sit at *different* cache positions (``pos`` [B]).

    Equivalent to ``model.decode_step`` when all positions agree; used only after a candidate backtracked, so that the
    other candidates of the batch keep decoding in the same forward pass.
    """
    cfg = model.cfg
    B = tok.shape[0]
    ct = cur_time.view(-1, 1)
    phase = torch.remainder(ct, cfg.n_phase)
    beat = (torch.div(ct, TPB, rounding_mode="floor") + cfg.beat_offset).clamp(0, cfg.n_beat - 1)
    x = (model.tok_emb(tok.view(-1, 1)) + model.pos_emb(pos.view(-1, 1))
         + model.phase_emb(phase) + model.beat_emb(beat))
    if cfg.lv_cont:
        x = x + model.lv_term(tok.view(-1, 1))
    if cache.g is not None:
        x = x + cache.g[:, None]
    dpos, dneg = model.deltas(ct, x)
    last = int(pos.max()) + 1 if last is None else last
    mask = (torch.arange(last, device=tok.device)[None, :] <= pos[:, None])[:, None, None, :]
    rows = torch.arange(B, device=tok.device)
    for l, blk in enumerate(model.dec_blocks):
        q, k, v = blk.sa._qkv(blk.ln1(x))
        cache.k[l][rows, :, pos] = k[:, :, 0].to(cache.k[l].dtype)
        cache.v[l][rows, :, pos] = v[:, :, 0].to(cache.v[l].dtype)
        y = F.scaled_dot_product_attention(q, cache.k[l][:, :, :last].to(q.dtype),
                                           cache.v[l][:, :, :last].to(q.dtype), attn_mask=mask)
        x = x + blk.sa.proj(y.transpose(1, 2).reshape(B, 1, -1))
        x = x + blk.ca(blk.ln2(x), cache.ck[l], cache.cv[l], dpos, dneg)
        if blk.sca is not None and cache.sk is not None:
            x = x + blk.sca(blk.ln_s(x), cache.sk[l], cache.sv[l], cache.smask, cache.skeep)
        x = x + blk.ff(blk.ln3(x))
    return F.linear(model.dec_ln(x[:, 0]), model.tok_emb.weight)


def replicate_cache(cache, copies):
    """Tile an R-row decode cache to ``copies * R`` rows (row k*R + r is a copy of row r)."""
    rep = lambda tensors: [t.repeat(copies, *([1] * (t.dim() - 1))) for t in tensors]
    out = DecodeCache(rep(cache.k), rep(cache.v), rep(cache.ck), rep(cache.cv))
    out.len = cache.len
    if cache.g is not None:
        out.g = cache.g.repeat(copies, 1)
    if getattr(cache, "sk", None) is not None:
        out.sk = [None if t is None else t.repeat(copies, 1, 1, 1) for t in cache.sk]
        out.sv = [None if t is None else t.repeat(copies, 1, 1, 1) for t in cache.sv]
        out.smask = cache.smask.repeat(copies, 1, 1, 1)
        out.skeep = cache.skeep.repeat(copies)
    return out


class _Candidate:
    """Decoding state of one window candidate (grammar, hand state, backtracking stack, token lists)."""
    def __init__(self, index, guard, rng_factory, V, options, max_tick, limit, bars=None):
        self.bars = bars                    # seq_version 6: the 4 forced BDI ids (None -> the old, bar-less grammar)
        self.index, self.guard, self.rng_factory = index, guard, rng_factory
        self.V, self.options, self.max_tick, self.limit = V, options, max_tick, limit
        self.snapshot = guard.clone()
        self.attempt = 0
        self.resamples = 0
        self.fallback = False
        self.total_backtracks = 0
        self.forced = False
        self.logits0 = None
        self.reset()

    def reset(self):
        o = self.options
        self.rng = self.rng_factory(self.index, self.attempt)
        self.grammar = EventGrammar(self.V, touch=o.touch, ex=o.ex, fest=o.fest, max_tick=self.max_tick,
                                    max_segments=o.max_segments, chain_groups=False, slide_mask=o.slide_mask, bars=self.bars)
        self.tokens, self.log_probs, self.stack = [], [], []
        self.banned = set()
        self.backtracks = 0
        self.iterations = 0
        self.safe_len = 0
        self.finished = False
        self.logits = self.logits0

    def abort(self, can_resample):
        """Budget exhausted: resample with a fresh seed, else keep the notes completed so far."""
        if can_resample and self.attempt < self.options.max_resamples:
            self.total_backtracks += self.backtracks
            self.attempt += 1
            self.resamples += 1
            self.guard.restore(self.snapshot)
            self.reset()                   # the driver rewinds this candidate's cache position to the prefix
            return
        self.total_backtracks += self.backtracks
        self.fallback = True
        keep = self.safe_len
        self.tokens = self.tokens[:keep] + [self.V.EOS]
        self.log_probs = self.log_probs[:keep] + [math.log(1e-3)]
        self.finished = True


class LegacyBackend:
    """One window = one private KV cache; steps run exactly as in the original batch-per-window implementation."""
    def __init__(self, model, ac=nullcontext):
        self.model, self.ac = model, ac
        self._pin = {}

    def begin_window(self, enc, prefixes, K, song=None):
        model, device = self.model, enc.device
        R = len(prefixes)
        tok = torch.as_tensor(np.stack([p["tokens"] for p in prefixes]), dtype=torch.long, device=device)
        ct = torch.as_tensor(np.stack([p["cur_time"] for p in prefixes]), dtype=torch.long, device=device)
        # speed: ONE autocast scope for the whole window (prefill + every decode step) so autocast's weight-cast cache survives
        # between steps (it is cleared on scope exit); the cached cast is the very same tensor a fresh cast would produce.
        stack = ExitStack()
        try:
            stack.enter_context(self.ac())
            cache = model.init_cache(enc.expand(R, -1, -1), song=song)
            logits0 = model.prefill(tok, ct, cache).float().cpu().numpy()
        except BaseException:
            stack.close()
            raise
        prefix_len = cache.len
        if K > 1:
            cache = replicate_cache(cache, K)
        handle = _Window(self, device, logits0, prefix_len, K, R, cache=cache)
        handle.stack = stack
        return handle

    def _pinned(self, key, shape, dtype):
        buf = self._pin.get(key)
        if buf is None:
            buf = self._pin[key] = torch.empty(shape, dtype=dtype, pin_memory=True)
        return buf

    def step(self, handle, tok, ct, pos, uniform):
        model, device = self.model, handle.device
        fast = device.type == "cuda" and getattr(handle, "stack", None) is not None
        if fast:
            # speed: one pinned async host->device copy for (tok, ct, pos) instead of three synchronous pageable copies
            n = len(tok)
            host = self._pinned(("in", n), (3, n), torch.int64)
            hv = host.numpy()
            hv[0] = tok
            hv[1] = ct
            hv[2] = pos
            dev_in = host.to(device, non_blocking=True)
            tok_t, ct_t, pos_t = dev_in[0], dev_in[1], dev_in[2]
            if uniform is not None and hasattr(model, "decode_step"):
                handle.cache.len = uniform
                out = model.decode_step(tok_t, ct_t, handle.cache)
            else:
                out = rows_step(model, tok_t, ct_t, handle.cache, pos_t)
            out = out.float()
            dst = self._pinned(("out", tuple(out.shape)), tuple(out.shape), torch.float32)
            dst.copy_(out, non_blocking=True)
            # blocking-sync event: the host thread sleeps instead of spin-waiting on the GPU (frees the CPU for other workers)
            ev = torch.cuda.Event(blocking=True)
            ev.record()
            ev.synchronize()
            return dst.numpy().copy()
        tok_t = torch.as_tensor(tok).to(device)
        ct_t = torch.as_tensor(ct).to(device)
        with self.ac():
            if uniform is not None and hasattr(model, "decode_step"):
                handle.cache.len = uniform
                out = model.decode_step(tok_t, ct_t, handle.cache)
            else:
                out = rows_step(model, tok_t, ct_t, handle.cache, torch.as_tensor(pos).to(device))
        return out.float().cpu().numpy()

    def end_window(self, handle):
        handle.cache = None
        stack = getattr(handle, "stack", None)
        if stack is not None:
            handle.stack = None
            stack.close()


class _Window:
    """Handle of one window's decode state (``logits0`` = prefill logits for the R prefix rows)."""
    def __init__(self, backend, device, logits0, prefix_len, K, R, cache=None, row0=0):
        self.backend, self.device, self.logits0, self.prefix_len = backend, device, logits0, prefix_len
        self.K, self.R, self.cache, self.row0 = K, R, cache, row0
        self.rows = K * R
        self.pad = 0


def run_sync(gen):
    """Drive a step-yielding generator to completion with the backend carried by its requests."""
    try:
        request = next(gen)
        while True:
            out = request[0].backend.step(*request)
            request = gen.send(out)
    except StopIteration as stop:
        return stop.value


def decode_candidates_co(model, enc, prefixes, options, V, guards, rng_factory, max_tick, *, bias=None, backend, song=None, bars=None):
    """Generator form of ``decode_candidates``: yields ``(window, tok[K*R], cur_time[K*R], pos[K*R], uniform_pos)``
    step requests and receives the float32 logits ``[K*R, V]``; returns the per-candidate results.  ``backend`` owns
    the KV cache (private per window, or rows of a pool shared with other songs)."""
    K, R = len(guards), len(prefixes)
    plan = guidance_plan(options)
    n_designers = designer_rows(options, R)
    window = backend.begin_window(enc, prefixes, K, song=song)
    try:
        logits0, prefix_len = window.logits0, window.prefix_len
        limit = min(options.max_target_tokens + 1, model.cfg.max_len - prefix_len)
        cands = [_Candidate(k, guards[k], rng_factory, V, options, max_tick, limit, bars=bars) for k in range(K)]
        for c in cands:
            c.logits0 = c.logits = logits0
        pos = [prefix_len] * K
        deadline = time.monotonic() + options.window_seconds
        max_iterations = 6 * limit + 8 * options.max_backtracks + 200
        pad = V.PAD
        timing = timing_mask(V)
        while True:
            feeds = {}
            timed_out = time.monotonic() > deadline
            for c in cands:
                if c.finished:
                    continue
                token = _advance(c, V, options, plan, n_designers, bias, timing, timed_out, max_iterations)
                if c.finished:
                    continue
                pos[c.index] = prefix_len + len(c.tokens) - (1 if token is not None else 0)
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
                    ct_row[base:base + R] = c.grammar.feed_time
            out = yield (window, tok_row, ct_row, pos_row, next(iter(positions)) if len(positions) == 1 else None)
            for k in feeds:
                cands[k].logits = out[k * R:(k + 1) * R]
                pos[k] += 1
        results = []
        for c in cands:
            stats = dict(tokens=len(c.tokens), prefix_tokens=prefix_len,
                         backtracks=c.total_backtracks + c.backtracks, token_budget_forced=c.forced,
                         resamples=c.resamples, fallback=c.fallback)
            results.append((c.tokens, c.log_probs, stats))
        return results
    finally:
        backend.end_window(window)


def decode_candidates(model, enc, prefixes, options, V, guards, rng_factory, max_tick, *, bias=None, song=None):
    """Decode ``len(guards)`` candidates of one window in a single batched pass.

    Every candidate has its own grammar, hand guard, RNG and backtracking stack; all share the prefixes (designer /
    guidance rows).  A candidate that exceeds ``options.max_backtracks`` (or the window time cap) is resampled with a new
    seed up to ``options.max_resamples`` times and finally truncated to its last completed note, so decoding always
    terminates.  Returns a list of ``(tokens, log_probs, stats)`` aligned with ``guards`` (each guard holds the hand
    state after its candidate's committed notes).
    """
    return run_sync(decode_candidates_co(model, enc, prefixes, options, V, guards, rng_factory, max_tick,
                                         bias=bias, backend=LegacyBackend(model), song=song))


def _advance(c, V, options, plan, n_designers, bias, timing, timed_out, max_iterations):
    """Run one candidate until it has a token to feed (returned) or is finished / reset (None)."""
    grammar, guard = c.grammar, c.guard
    while True:
        if grammar.finished:
            c.finished = True
            return None
        c.iterations += 1
        if timed_out or c.iterations > max_iterations or c.backtracks > options.max_backtracks:
            c.abort(can_resample=not timed_out)
            return None
        remaining = c.limit - len(c.tokens)
        allowed = np.asarray(guard.allowed(grammar, remaining), np.int64)
        if c.banned:
            allowed = allowed[~np.isin(allowed, np.fromiter(c.banned, np.int64, len(c.banned)))]
        if not len(allowed):
            if not c.stack:
                c.abort(can_resample=True)       # dead end at the start of a note: nothing to roll back
                return None
            c.grammar, c.logits, banned, token = c.stack.pop()
            grammar = c.grammar
            c.tokens.pop()
            c.log_probs.pop()
            c.banned = set(banned)
            c.banned.add(token)
            c.backtracks += 1
            continue
        step_bias = bias(grammar, guard) if bias is not None else None
        token, probability = sample_token(c.logits, allowed, options, c.rng, n_designers, plan, step_bias, timing, grammar)
        name = V.toks[token]
        boundary = token == V.EOS or V.is_time(token) or name.startswith("N_") or token == getattr(V, "barm", None)
        if boundary:
            before_commit = guard.committed_count
            guard.commit(grammar)
            if guard.committed_count != before_commit:
                c.stack.clear()  # completed notes never need graph rollback
                c.safe_len = len(c.tokens)
        c.stack.append((grammar.clone(), c.logits, c.banned.copy(), token))
        if remaining < 12:
            c.forced = True
        grammar.accept(token)
        c.tokens.append(token)
        c.log_probs.append(probability)
        c.banned = set()
        if grammar.finished:
            c.finished = True
            return None
        return token


def decode_window(model, enc, prefixes, options, V, guard, rng, max_tick, bias=None):
    """Single-candidate decode (legacy entry point); ``guard`` is updated in place with the committed notes."""
    def rng_factory(index, attempt):
        return rng if attempt == 0 else np.random.default_rng(int(rng.integers(0, 2**32)) + attempt)
    (tokens, log_probs, stats), = decode_candidates(model, enc, prefixes, options, V, [guard], rng_factory,
                                                    max_tick, bias=bias)
    return tokens, log_probs, stats



def candidate_score(playability, estimated_level, target_level, mean_log_prob):
    """Lexicographic priority: playability, requested level error, likelihood."""
    return (int(playability["n_viol"]),
            abs(float(estimated_level) - target_level) if target_level is not None else 0.0,
            -float(mean_log_prob))


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _metadata_text(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def write_output(out, audio_path, notes, grid, options, info, *, title, artist, designers):
    """Stage, reparse and compare notes before publishing the three local files."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = ("maidata.txt", "track.mp3", "gen_info.json")
    if any((out / name).exists() for name in targets):
        raise FileExistsError(f"output already contains a chart: {out}; choose a new --out")
    slot = 5 if options.slot == "master" else 6
    estimate = info["level_estimate"]
    level = parse_level(options.level)
    level = estimate if level is None else level
    style = ", ".join(name for _, name in designers if name not in ("UNK", "OTHER"))
    tm = simai.TempoMap(list(zip(grid["beats"], grid["bpms"])), grid["first"])
    text = (f"&title={_metadata_text(title)}\n&artist={_metadata_text(artist)}\n"
            f"&first={grid['first']:.10g}\n&wholebpm={grid['bpms'][0]:.10g}\n"
            f"&lv_{slot}={level:.1f}\n&des_{slot}=maiAI{(' / ' + style) if style else ''}\n"
            f"&inote_{slot}={simai.write_chart(notes, tm)}\n")
    with tempfile.TemporaryDirectory(prefix=".generate-", dir=out) as stage_dir:
        stage = Path(stage_dir)
        (stage / "maidata.txt").write_text(text, encoding="utf-8")
        meta, charts = simai.load_song(stage)
        parsed, parsed_tm = charts[slot]
        quantized, _ = simai.quantize(parsed)
        expected = [simai.canon_key(n) for n in canonicalize(notes)]
        actual = [simai.canon_key(n) for n in canonicalize(quantized)]
        if actual != expected:
            raise RuntimeError("written maidata did not round-trip to the generated events")
        if not np.allclose(parsed_tm.beats, tm.beats, atol=0.51 / TPB) or not np.allclose(parsed_tm.bpms, tm.bpms):
            raise RuntimeError("written maidata changed the beat grid")
        audio_path = Path(audio_path).resolve()
        if audio_path.suffix.casefold() == ".mp3":
            shutil.copyfile(audio_path, stage / "track.mp3")
        else:
            import subprocess
            subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(audio_path),
                            "-vn", "-codec:a", "libmp3lame", "-b:a", "320k", str(stage / "track.mp3")],
                           check=True, capture_output=True)
        info["round_trip_valid"] = True
        info["output"] = str(out)
        (stage / "gen_info.json").write_text(
            json.dumps(json_value(info), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in targets:
            (stage / name).rename(out / name)
    return out


class ChartGenerator:
    feat_cfg = None          # feature-fusion checkpoints (v3a): dict(encoder, layer, dim, fps); None = mel-only model
    _feat_enc = None
    _song_sid = {}
    infill_gen = None        # optional second ChartGenerator with the FUT infill model (attached by tools/generate.py)

    def __init__(self, checkpoint=ROOT / "work/runs/v1/best.pt", *, device="cuda:1",
                 meta=ROOT / "work/meta.json", amp="auto"):
        started = time.perf_counter()
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        ckpt, self.checkpoint_info = stable_checkpoint(checkpoint)
        with Path(meta).open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        names = [entry if isinstance(entry, str) else entry["name"] for entry in metadata["designers"]]
        if ckpt.get("designers") is not None and list(ckpt["designers"]) != names:
            raise ValueError("checkpoint designer vocabulary does not match metadata")
        raw_cfg = ckpt["cfg"]
        cfg = ModelConfig.from_dict(raw_cfg)
        if cfg.seq_version not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"unsupported checkpoint seq_version={cfg.seq_version}; expected 1 .. 6")
        self.V = Vocab(names, seq_version=cfg.seq_version, with_fut=bool(getattr(cfg, "with_fut", False)))
        if cfg.vocab_size != self.V.size:
            raise ValueError("checkpoint vocabulary size does not match metadata")
        if cfg.n_cond != self.V.n_cond:
            raise ValueError("checkpoint n_cond does not match its seq_version condition header")
        if cfg.lv_cont and (cfg.lv0 != self.V["LV_0"] or cfg.n_lv != 141):
            raise ValueError("checkpoint continuous-level vocabulary does not match its LV tokens")
        self.seq_meta = ckpt.get("seq_meta")
        self.v3_stats = None
        if cfg.seq_version >= 3:
            if not cfg.song_dim or not isinstance((self.seq_meta or {}).get("v3"), dict):
                raise ValueError("seq_version=3 checkpoint requires cfg.song_dim > 0 and seq_meta['v3'] (plan / REF statistics)")
            if not cfg.feat_encoder:
                raise ValueError("seq_version=3 checkpoint without MERT feature fusion is not supported by the generator")
            self.v3_stats = self.seq_meta["v3"]
        if cfg.seq_version >= 4 and not getattr(cfg, "song_ann", False):
            raise ValueError("seq_version=4 checkpoint without cfg.song_ann is inconsistent")
        if cfg.seq_version >= 2:
            _validate_v2_density_meta(self.seq_meta)
            self.checkpoint_info.update(seq_version=cfg.seq_version, seq_meta=self.seq_meta)
        self.model = ChartModel(cfg)
        self.feat_cfg = (dict(encoder=cfg.feat_encoder, layer=int(cfg.feat_layer), dim=int(cfg.feat_dim), fps=float(cfg.feat_fps))
                         if cfg.feat_encoder else None)
        if self.feat_cfg:
            self.checkpoint_info.update(feat_encoder=self.feat_cfg)
        self._feat_enc = None
        self._song_sid = {s[1]: int(s[0]) for s in metadata.get("songs", [])}
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device).eval()
        del ckpt
        if amp == "auto":
            # bf16 on Ampere+, fp16 on tensor-core GPUs (Volta/Turing); Pascal has no fast fp16, fp32 is faster there
            major = torch.cuda.get_device_capability(self.device)[0] if self.device.type == "cuda" else 0
            amp = "bf16" if major >= 8 else ("fp16" if major >= 7 else "none")
        if amp not in ("none", "fp16", "bf16"):
            raise ValueError("amp must be auto, none, fp16 or bf16")
        if amp == "bf16" and (self.device.type != "cuda"
                              or torch.cuda.get_device_capability(self.device)[0] < 8):
            raise ValueError("bf16 requires an Ampere-or-newer CUDA device; use --amp fp16 or none")
        self.amp = amp
        self.dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
        self.load_seconds = time.perf_counter() - started

    def autocast(self):
        return (torch.autocast(self.device.type, dtype=self.dtype)
                if self.dtype is not None else nullcontext())

    @torch.inference_mode()
    def generate(self, audio_path, options=None, *, out=None, title=None, artist="", log=print):
        """Generate one chart (private per-window KV caches; the reference, bit-reproducible path)."""
        return run_sync(self.generate_co(audio_path, options, out=out, title=title, artist=artist, log=log,
                                         backend=LegacyBackend(self.model, self.autocast)))

    def free_feat_encoder(self):
        self._feat_enc = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def song_feats(self, audio_path, wave, options, log=print):
        """Frozen-encoder features [T, D] fp16 (75 Hz for MERT) of the decoded track, or (None, info) for a mel-only run.
        auto: precomputed dataset / test-song file (verified bit-identical to extraction) -> cache -> extract and cache."""
        fc = self.feat_cfg
        t0 = time.perf_counter()
        if fc is None or options.feats == "off":
            return None, dict(source="off" if fc else "none")
        n_expected = len(wave) // 320 + 1 if fc["encoder"].startswith("mert") else None
        enc, layer = fc["encoder"], fc["layer"]

        def usable(arr):
            return arr.ndim == 2 and arr.shape[1] == fc["dim"] and (n_expected is None or arr.shape[0] == n_expected)

        if options.feats != "auto":
            arr = np.load(options.feats, mmap_mode="r")
            if not usable(arr):
                raise ValueError(f"feature file {options.feats} has shape {arr.shape}, expected [{n_expected}, {fc['dim']}]")
            return arr, dict(source="file", path=str(options.feats), seconds=time.perf_counter() - t0)
        cands = []
        sid = self._song_sid.get(audio_path.parent.name) if audio_path.name == "track.mp3" else None
        if sid is not None:
            cands.append(("dataset", feats_mod.feats_path(enc, layer, sid)))
        cands.append(("test_song", feats_mod.feats_path(enc, layer, "ext_" + audio_path.stem)))
        for source, path in cands:
            if os.path.isfile(path):
                try:
                    arr = np.load(path, mmap_mode="r")
                except Exception:
                    continue
                if usable(arr):
                    return arr, dict(source=source, path=path, seconds=time.perf_counter() - t0)
        digest = hashlib.sha1()
        with audio_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                digest.update(block)
        cache_dir = Path(os.environ.get("MAI_FEATS_CACHE") or (ROOT / "work" / "feats_cache")) / enc / f"L{layer}"
        cache = cache_dir / (digest.hexdigest()[:16] + ".npy")
        if cache.is_file():
            try:
                arr = np.load(cache, mmap_mode="r")
                if usable(arr):
                    return arr, dict(source="cache", path=str(cache), seconds=time.perf_counter() - t0)
            except Exception:
                pass
        import sys
        tools_dir = str(ROOT / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf_cache"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import extract_feats as E
        t_load = time.perf_counter()
        if self._feat_enc is None:
            self._feat_enc = E.load_encoder(enc, str(self.device))
        t_ext = time.perf_counter()
        arr = E.extract(wave, self._feat_enc, [layer])[layer]
        extract_seconds = time.perf_counter() - t_ext
        if not usable(arr):
            raise RuntimeError(f"extracted features have shape {arr.shape}, expected [{n_expected}, {fc['dim']}]")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.stem + f".tmp{os.getpid()}.npy")
            np.save(tmp, arr)
            os.replace(tmp, cache)
        except OSError as exc:
            log(f"warning: feature cache write failed: {exc}", flush=True)
        log(f"{enc} L{layer} features extracted in {extract_seconds:.1f} s (encoder load {t_ext - t_load:.1f} s)", flush=True)
        return arr, dict(source="extracted", cache=str(cache), seconds=time.perf_counter() - t0,
                         extract_seconds=extract_seconds, load_seconds=t_ext - t_load)

    def encode_window(self, mel, tempo, first, start, mean, std, feat_arr=None):
        """encoder output [1, 432, d] of the window starting at tick ``start``: log-mel (+ beat-synced frozen features, the same
        time grid as the mel, exactly as in training) -> ChartModel.encode"""
        times_out = []
        features = window_mel(mel, tempo, first, start, mean, std, out_times=times_out)
        fe = None
        if feat_arr is not None:
            sl, i0, fw, fvalid = feats_mod.window_slice_feats(feat_arr, self.feat_cfg["fps"], times_out[0])
            fe = dict(fsl=torch.from_numpy(np.ascontiguousarray(sl))[None], fi0=torch.from_numpy(i0)[None],
                      fw=torch.from_numpy(fw)[None], fvalid=torch.from_numpy(fvalid)[None])
        mel_t = torch.as_tensor(features[None], device=self.device)
        with self.autocast():
            return self.model.encode(mel_t) if fe is None else self.model.encode(mel_t, fe)

    def v3_context(self, mel, mean, std, feat_arr, tempo, first, duration, options, log=print):
        """seq_version 3, once per song: bar table on the generation beat grid (same code as the training tables) -> song encoder pass ->
        PLAN per window.  Returns (SongContext | None, info dict).  No MERT features -> everything off (the song path was trained with MERT)."""
        t0 = time.perf_counter()
        _plan_unk_env = os.environ.get("MAI_PLAN_UNK", "").strip() not in ("", "0")     # v11 hook (default off): every window gets PLAN_UNK
        if _plan_unk_env:
            options.v3_plan = False
        if not (options.v3_song or options.v3_ref or options.v3_plan):
            return None, dict(enabled=False, reason="v3_song, v3_ref and v3_plan all off")
        if feat_arr is None:
            log("warning: seq_version 3 without MERT features: song context, REF and PLAN are switched off", flush=True)
            return None, dict(enabled=False, reason="no MERT features")
        n = V3.n_bars_of(tempo, first, duration, 0)
        tab = V3.compute_bar_table(mel, mean, std, feat_arr, float(self.feat_cfg["fps"]), tempo, first, n)
        t1 = time.perf_counter()
        ann_rec, ann_info = self.ann_record(options, tempo, first, n, duration, V3.bar_embed(tab), log=log, tab=tab) \
            if self.V.seq_version >= 4 else (None, None)
        t_ann = time.perf_counter()
        ctx = v3_infer.SongContext(self.model, self.V, tab, tempo, self.v3_stats, self.device, plan_mode=options.plan_mode,
                                   ann=ann_rec)
        t2 = time.perf_counter()
        info = dict(enabled=True, n_bars=int(n), pool_factor=int(ctx.g), table_seconds=round(t1 - t0, 3),
                    encoder_seconds=round(t2 - t_ann, 3), plan_mode=options.plan_mode,
                    plan_buckets=[int(x) for x in ctx.plans[::4]])
        if ann_info is not None:
            info["ann"] = dict(ann_info, load_seconds=round(t_ann - t1, 3))
        if _plan_unk_env:
            info["plan_unk_env"] = True
            info["plan_buckets_model"] = [int(x) for x in ctx.plans_model[::4]]
        if getattr(ctx, "plan_override_info", None):
            info["plan_override"] = ctx.plan_override_info
        if getattr(ctx, "bd_override_info", None):
            info["bd_override"] = ctx.bd_override_info
        info["v3_plan_effective"] = bool(options.v3_plan)
        return ctx, info

    def ann_record(self, options, tempo, first, nb, duration, Bn=None, log=print, tab=None):
        """seq_version 4: --ann spec -> (mai.ann.AnnRecord | None on the generation bar grid, info dict).
        The record dict carries section start bars / label ids / moment bars on the grid of THIS generation (nb bars); a record stored on another
        grid is rescaled.  The raw LLM answer form ({sections, moments}) is parsed and snapped like tools/prep_ann.py."""
        from . import ann as A
        spec = (options.ann or "none").strip()
        if spec == "none":
            return None, dict(mode="none", record=False, note="every bar / window token is UNK")
        if spec == "auto":
            log("warning: --ann auto is reserved (whole-song planner); using no annotation (all UNK)", flush=True)
            return None, dict(mode="auto", record=False, note="auto not implemented yet -> UNK")
        if spec == "pred":
            if tab is None:
                log("warning: --ann pred needs the bar table; using no annotation (all UNK)", flush=True)
                return None, dict(mode="pred", record=False, note="no bar table -> UNK")
            raw = plan_mod.predicted_record(tab, nb)
            rec = A.AnnRecord.from_dict(dict(nb=int(raw.get("nb", nb)), starts=raw["starts"], labels=raw["labels"],
                                             mbar=raw.get("mbar", ()), mkind=raw.get("mkind", ())))
            info = dict(mode="pred", form="record", source="scratch/plan/sectionpred.py", record=True,
                        n_sec=int(len(rec.starts)), labels=[A.LABELS[int(x)] for x in rec.labels], starts=[int(x) for x in rec.starts])
            return rec, info
        path = Path(spec[5:])
        if not path.is_file():
            raise FileNotFoundError(f"--ann file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "starts" in raw and "labels" in raw:                    # record form (v4 annotation JSON)
            rec = A.AnnRecord.from_dict(dict(nb=int(raw.get("nb", nb)), starts=raw["starts"], labels=raw["labels"],
                                             mbar=raw.get("mbar", ()), mkind=raw.get("mkind", ())))
            info = dict(mode="file", form="record", source=str(path), record=True)
            if rec.nb != nb:
                scale = nb / max(int(rec.nb), 1)
                starts = np.clip(np.round(np.asarray(rec.starts) * scale).astype(int), 0, nb - 1)
                starts[0] = 0
                starts = np.maximum.accumulate(starts)
                rec = A.AnnRecord(nb, starts, rec.labels, np.clip(np.round(np.asarray(rec.mbar) * scale).astype(int), 0, nb - 1),
                                  rec.mkind)
                info["rescaled_from"] = int(raw.get("nb", nb))
        else:                                                                                # raw answer form
            if isinstance(raw, dict):
                text = raw["content"] if "content" in raw else json.dumps(raw, ensure_ascii=False)
            else:
                text = str(raw)
            secs, moms, mode = A.parse_raw(text)
            if not secs:
                raise ValueError(f"--ann {path}: no parseable sections")
            rec, info = A.build_record(secs, moms, tempo, first, nb, duration, Bn)
            info = dict(info, mode="file", form="raw", parse=mode, source=str(path), record=rec is not None)
            if rec is None:
                log(f"warning: --ann {path} rejected ({info.get('reason')}); using no annotation (all UNK)", flush=True)
        if rec is not None:
            info.update(n_sec=int(len(rec.starts)),
                        labels=[A.LABELS[int(x)] for x in rec.labels],
                        starts=[int(x) for x in rec.starts])
        return rec, info

    def build_prefixes(self, packed, start, rows, bpm_token, prog_token, reservation, ref, log=print):
        """one token prefix per condition row.  ``ref`` (seq_version 3) = (marker, tokens, cur_time) -> (prefixes, ref actually used).
        The REF events count against ``max_len``: when the reserved target leaves no room the REF segment is cut at a note
        boundary (or dropped to REF_NONE) instead of failing."""
        V, max_len = self.V, self.model.cfg.max_len
        kw = dict(max_len=max_len, include_target=False, reserve_target_tokens=reservation)
        prefixes = [build_window_tokens(V, *packed, start, row, bpm_token, prog_token, ref=ref, **kw) for row in rows]
        if prefixes[0] is None and ref is not None and ref[1]:
            return self.build_prefixes(packed, start, rows, bpm_token, prog_token, reservation, self.shrink_ref(ref, rows, reservation))
        return prefixes, ref

    def shrink_ref(self, ref, rows, reservation):
        """largest note-boundary prefix of the REF events that fits next to the reserved target (REF_NONE if nothing fits)"""
        room = self.model.cfg.max_len - (len(rows[0]) + 5) - 1 - int(reservation) - 1 - 48      # keep a little room for history
        toks, cts = list(ref[1]), list(ref[2])
        # drop whole trailing notes (a note starts at a token that is a time token or an N_ kind token following one)
        while toks and len(toks) > max(room, 0):
            cut = len(toks) - 1
            while cut > 0 and not self.V.toks[int(toks[cut])].startswith("N_"):
                cut -= 1
            toks, cts = toks[:cut], cts[:cut]
            while toks and self.V.is_time(int(toks[-1])):
                toks, cts = toks[:-1], cts[:-1]
        if not toks:
            return (self.V["REF_NONE"], [], [])
        return (ref[0], toks, cts)

    def _history_notes(self, tokens):
        """note-kind tokens between CTX and TGT of a prefix (the REF events in front of CTX are not history)"""
        V = self.V
        tokens = [int(t) for t in tokens]
        start = tokens.index(V.CTX) + 1 if V.CTX in tokens else 0
        return sum(V.toks[t].startswith("N_") for t in tokens[start:])

    def plan_ratio(self, bucket):
        """typical window / song-mean density ratio of a PLAN bucket (centre of the equal-mass bucket on the training quantile function)"""
        q = self.v3_stats.get("plan_q")
        n = V3.N_PLAN
        if q is None:
            e = np.asarray(self.v3_stats["plan_edges"], np.float64)
            lo = np.concatenate([[e[0] * 0.7], e]); hi = np.concatenate([e, [e[-1] * 1.3]])
            return float((lo[bucket] + hi[bucket]) / 2)
        return float(np.interp((int(bucket) + 0.5) / n, np.linspace(0.0, 1.0, len(q)), np.asarray(q, np.float64)))

    def rows_needed(self, options):
        """Decoder rows one song of ``options`` occupies in a shared pool: window candidates x condition rows."""
        designers = resolve_designers(options.designer, self.V.designers)
        if parse_level(options.level) is None and options.auto_level:
            options = replace(options, level=f"{AUTO_LEVEL_MEAN:.1f}")      # the level is predicted later; same row layout
        K = options.window_candidates
        if float(getattr(options, "chorus", 0.0) or 0.0) > 0 and "pool" in chorus_mod.parse_parts(options.chorus_parts):
            K = max(K, int(round(K * (1.0 + max(0.0, float(options.chorus_k))))))   # pool: chorus windows ask for more rows
        return K * n_condition_rows(options, len(designers))

    def generate_co(self, audio_path, options=None, *, out=None, title=None, artist="", log=print, backend):
        """Generator form of ``generate``: yields decoder step requests (see ``decode_candidates_co``) and returns the
        info dict.  Must be advanced inside ``torch.inference_mode()``; model calls enter autocast themselves."""
        options = options or GenerationOptions()
        options.validate()
        options = qmode.resolve_quality(options)
        quality_on = options.quality != "fast"
        infill_on = bool(options.infill_passes > 0 and getattr(self, "infill_gen", None) is not None)
        if quality_on and not isinstance(backend, LegacyBackend):
            raise ValueError("--quality fine / max needs the private-cache (sequential) backend, not the pooled one")
        started = time.perf_counter()
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        requested = options                  # recorded in gen_info; ``options`` below carries the effective (auto) level
        designers = resolve_designers(options.designer, self.V.designers)
        version = resolve_version(options.version)
        grid = resolve_grid(audio_path, options, self.device)
        tempo = NpTempo(grid["beats"], grid["bpms"])
        first = float(grid["first"])
        grid_seconds = time.perf_counter() - started
        wave = audio.load_audio(audio_path)
        duration = len(wave) / audio.SR
        if not duration > first:
            raise ValueError("beat-grid first is at or after audio end")
        mel = audio.mel_db(wave, device=str(self.device))
        mel32 = mel.astype(np.float32)
        mean, std = float(mel32.mean()), float(mel32.std())
        del mel32
        feat_arr, feat_info = self.song_feats(audio_path, wave, options, log=log)
        del wave
        auto_level = None
        if parse_level(options.level) is None and options.auto_level:
            auto_level = auto_level_from_audio(mel, grid, options, log=log)
            options = replace(options, level=f"{auto_level['level']:.1f}")
        rows = make_conditions(self.V, options, designers, version, getattr(self, "seq_meta", None))
        v3ctx, v3_info = (self.v3_context(mel, mean, std, feat_arr, tempo, first, duration, options, log=log)
                          if self.V.seq_version >= 3 else (None, None))
        song_beats = float(tempo.to_beat(duration - first))
        end_tick = max(1, int(math.ceil(song_beats * TPB)))
        starts = list(range(0, end_tick, T_MAX))
        if len(grid["bpms"]) > 1:
            log(f"tempo map: {len(grid['bpms'])} segments, "
                f"{_window_seconds(grid, 0, end_tick):.1f} s of chart "
                f"({', '.join(f'{float(bpm):g}' for bpm in grid['bpms'])})", flush=True)
        if options.max_windows:
            starts = starts[:options.max_windows]
        if not starts:
            raise ValueError("audio has no generation windows")
        slot = 5 if options.slot == "master" else 6
        target_level = parse_level(options.level)
        curve = density = None
        if options.window_candidates > 1 or target_level is not None:
            curve = activity.ActivityCurve(mel, tempo, first, song_beats=song_beats)
            if target_level is not None:
                try:
                    density = activity.get_density_model()
                except OSError:
                    log("warning: work/density_model.json missing; density term disabled", flush=True)
        weights = rerank.RerankWeights(logprob=options.w_logprob, activity=options.w_activity,
                                       repeat=options.w_repeat, density=options.w_density, intensity=options.w_intensity,
                                       pair=options.repeat_pair, far=options.repeat_far, band=options.density_band,
                                       guard_empty=options.guard_empty if options.guard else 0,
                                       guard_run=4 if options.guard else 0,
                                       guard_floor=options.guard_floor if options.guard else 0.0,
                                       guard_streak=options.guard_streak if options.guard else 0.0,
                                       guard_play_db=options.guard_play_db)
        expect = None
        plan_state = plan_mod.build(self, v3ctx, target_level, options, len(starts), log=log) if options.plan > 0 else None
        if options.expect_mode == "level" and target_level is not None and curve is not None and density is not None:
            plan_ratios = None
            if v3ctx is not None and options.v3_plan:
                plan_ratios = [self.plan_ratio(v3ctx.plan_bucket(st // V3.BAR, options.plan_shift)) for st in starts]
            expect_params = None
            if plan_state is not None and "dens" in plan_state.parts:
                plan_ratios, expect_params = plan_state.density_inputs(self, v3ctx, starts, plan_ratios)
            if plan_state is not None and "token" in plan_state.parts and options.v3_plan:
                plan_state.apply_tokens(self, v3ctx, starts, [self.plan_ratio(v3ctx.plan_bucket(st // V3.BAR, options.plan_shift)) for st in starts])
            expect = activity.WindowExpectation(
                curve, density, target_level, plan_ratios=plan_ratios, params=expect_params,
                scale=options.density_scale * (1.0 + options.expect_hi * max(0.0, target_level - DENSITY_KNEE)))
            log(f"expected notes: {expect.nps:.2f}/s at level {target_level:.1f}, {expect.bpm:.0f} bpm"
                f" ({'audio + PLAN' if expect.has_plan else 'audio only'}), {float(expect.expected[expect.ok].sum()):.0f} in total", flush=True)
        # chorus helper (mai/chorus.py): None when --chorus 0 -> every line below is inert
        chorus_state = chorus_mod.build(self, v3ctx, target_level, options, len(starts), log=log)
        if chorus_state is not None:
            chorus_state.attach_onset(mel, tempo, first, end_tick, log=log)     # rwo: audio onset gate (inert unless rwo)
        if chorus_state is not None and expect is not None:
            chorus_state.apply_dens(expect)
            if np.any(chorus_state.mults != 1.0):
                log(f"chorus dens: expected notes of {chorus_state.stats['dens_windows']} windows x "
                    f"{float(chorus_state.mults.max()):.3f} (max), song total kept "
                    f"({float(expect.expected[expect.ok].sum()):.0f})", flush=True)
        calibration = load_generation_calibration(self.V, options, log=log)
        profile = None
        if options.style_steer > 0:
            try:
                profile = steer_mod.get_style_profile()
            except OSError:
                log("warning: work/style_model.json missing; style steering disabled", flush=True)
        kind_prior = None
        if options.kind_prior > 0:
            from . import audiobars
            prior_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "audio_kind_prior.npz")
            kind_prior = audiobars.AudioKindPrior(prior_path).offsets(mel, feat_arr, tempo, first, int(math.ceil(song_beats / 4.0)) + 2,
                                                                       options.kind_prior)
        feature_seconds = time.perf_counter() - started - grid_seconds
        encoded = {}
        candidates = []
        K = options.window_candidates
        sm_tempo = simai.TempoMap(list(zip(grid["beats"], grid["bpms"])), grid["first"])

        def make_scorer():
            sc = rerank.WindowScorer(curve, density, target_level, weights,
                                     brk_timing=options.brk_timing_gate)
            sc.scale = options.density_scale * (1.0 + options.density_scale_hi * max(0.0, (target_level or 0.0) - DENSITY_KNEE))
            sc.expect = expect
            sc.chorus = chorus_state
            return sc

        def make_bias(sc):
            exp_fn = lambda tick, _s=sc: _s.expected_notes_table(tick) or 0.0
            if plan_state is not None and "dens" in plan_state.parts and sc.expect is not None:
                exp_fn = plan_state.expected_fn(sc, starts)          # the density feedback follows the planned window densities
            if chorus_state is not None:
                exp_fn = chorus_state.wrap_expected(exp_fn)          # dens: the density feedback follows the same chorus multiplier
            return Steer(self.V, options, level=target_level, profile=profile, calibration=calibration,
                         expected_fn=exp_fn,
                         intensity_fn=(lambda tick, _s=sc: _s.nps_profile(tick)) if options.act_prior > 0 else None,
                         intensity_mean=sc.mean_nps() if options.act_prior > 0 else None,
                         song_seconds=duration - first, n_windows=len(starts), kind_prior=kind_prior, plan=plan_state,
                         chorus=chorus_state)

        def make_gate():
            return murigate.MuriGate(sm_tempo, target_level, w_hard=options.muri_w_hard, w_soft=options.muri_w_soft,
                                     hard_level=options.muri_hard_level, slack=options.muri_slack, n_windows=len(starts), log=log)

        def encode(start_):
            if start_ not in encoded:
                encoded[start_] = self.encode_window(mel, tempo, first, start_, mean, std, feat_arr)
            return encoded[start_]

        def make_hands():
            if options.hand_mode == "travel":
                return handtravel.ChartHands(tempo)
            tgt = handtarget.target_for(target_level, options.hand_targets or None)
            return handtravel.ChartHands(tempo, budget=handtarget.HandBudget(
                tgt, n0=options.hand_n0, z0=options.hand_z0, kappa=options.hand_kappa, w_hard=options.hand_hard, w_cross=options.hand_cross_w))

        q = SimpleNamespace(gen=self, options=options, V=self.V, model=self.model, backend=backend, tempo=tempo, first=first,
                            end_tick=end_tick, starts=starts, song_beats=song_beats, v3ctx=v3ctx, rows=rows, encode=encode,
                            make_scorer=make_scorer, make_bias=make_bias, make_gate=make_gate, make_hands=make_hands, log=log, seed=0, scale0=1.0,
                            refine_gate=make_gate() if ((quality_on and (options.refine_passes > 0 or options.quality == "max")) or infill_on) else None, last_events=None,
                            selftest_done=False, mel=mel, mean=mean, std=std, feat_arr=feat_arr,
                            infill=self.infill_gen,
                            infill_encode=(lambda st: self.infill_gen.encode_window(mel, tempo, first, st, mean, std, feat_arr)
                                           if self.infill_gen is not None else None))
        q_time = dict(lookahead=0.0, refine=0.0, hand=0.0, lookahead_windows=0, lookahead_changed=0)

        def load_infill_records(src):
            """Build window records from an already generated chart, for infill-only refinement."""
            path = Path(src)
            if path.is_dir():
                path = path / "maidata.txt"
            if not path.is_file():
                raise FileNotFoundError(f"infill source chart not found: {path}")
            cmeta, chs = simai.read_maidata(str(path))
            slot_i = 5 if 5 in chs else max(chs)
            wbpm = float((cmeta.get("wholebpm", "") or "120").split("||")[0].strip() or 120)
            cfirst = float((cmeta.get("first", "") or "0").split("||")[0].strip() or 0)
            cnotes, _ = simai.parse_chart(chs[slot_i], wbpm, cfirst)
            cnotes = canonicalize(simai.quantize(cnotes)[0])
            # The current V6 gate forbids 4+ segment arms.  Old reference drafts were made
            # before that guard: cap their last over-long segment at three while preserving
            # the original per-segment timing.
            for n in cnotes:
                if n.kind != "slide":
                    continue
                for arm in n.arms:
                    if len(arm.segs) <= 3:
                        continue
                    total_dur = float(sum(d for _, d in arm.groups))
                    arm.segs = arm.segs[:3]
                    arm.groups = [(3, total_dur)]
            recs = []
            for j, st in enumerate(starts):
                en = min(int(st) + int(T_MAX), int(end_tick))
                ws = [n for n in cnotes if int(st) <= int(n.beat) < en]
                toks = encode_events(self.V, ws, int(st), 0, int(TGT))
                toks.append(self.V.EOS)
                stats = dict(chorus=bool(chorus_state is not None and chorus_state.is_chorus(j)),
                             chosen=-1, fallback=False, resamples=0, backtracks=0,
                             resamples_all=0, fallback_all=0)
                recs.append(dict(index=j, start=int(st), end=en, notes=list(ws), tokens=toks,
                                 probs=[], mean_lp=-0.5, scale=1.0, stats=stats,
                                 ref_block=None, ref_tokens=0))
            return recs

        for candidate_index in range(options.candidates):
            seed = (options.seed + candidate_index) % 2**32
            sequential_rng = np.random.default_rng(seed)
            scorer = make_scorer()
            scale0 = scorer.scale
            bias = make_bias(scorer)
            q.seed, q.scale0 = seed, scale0
            notes, probabilities, windows, records = [], [], [], []
            gate = make_gate() if options.muri else None
            hands = make_hands() if (options.w_hand > 0 or (quality_on and (options.refine_passes > 0 or options.la_top > 0))) else None
            tracker = playcheck.HandTracker(tempo)
            holds = []
            carry_end = 0
            previous_tokens = 384
            source_records = load_infill_records(options.infill_source) if (infill_on and options.infill_source) else None
            if source_records is not None:
                records = source_records
                notes = canonicalize([n for r in records for n in r["notes"]])
                windows = [r["stats"] for r in records]
            for index, start in enumerate(starts if source_records is None else []):
                packed = pack_notes(self.V, notes)
                bpm_token, prog_token = bpm_progress_tokens(self.V, tempo, start, song_beats)
                reservation = min(options.max_target_tokens, max(256, previous_tokens + 96),
                                  self.model.cfg.max_len - self.V.header_len - 2)
                w3, ref, rows_w = None, None, rows
                if self.V.seq_version >= 3:
                    ref = (self.V["REF_NONE"], [], [])
                    if v3ctx is not None:
                        # everything that depends on the notes generated so far: REF (retrieval) and the song memory of this window
                        assert start % V3.BAR == 0
                        w3 = v3ctx.window(start // V3.BAR, *packed, batch=len(rows), plan_shift=options.plan_shift,
                                          use_plan=options.v3_plan, use_ref=options.v3_ref, use_song=options.v3_song)
                        ref = w3["ref"]
                    rows_w = rows_with_window(self.V, options, rows, w3["plan_token"] if w3 else self.V["PLAN_UNK"],
                                              (w3 or {}).get("sec_token"), (w3 or {}).get("secn_token"), (w3 or {}).get("bd_tokens"))
                prefixes, ref_used = self.build_prefixes(packed, start, rows_w, bpm_token, prog_token, reservation, ref, log=log)
                if prefixes[0] is None:
                    raise ValueError("checkpoint max_len is too small for an inference prefix")
                if w3 is not None and options.plan_density > 0 and w3["plan"] is not None:
                    scorer.scale = scale0 * float(self.plan_ratio(w3["plan"])) ** options.plan_density
                base = HandGuard(self.V, tempo, start, end_tick, tracker, options.prune_end)
                base.holds = holds
                base.recent = [n for n in notes if n.beat >= start - 6 * 4 * TPB]
                window_end = min(start + T_MAX, end_tick)
                bias.begin_window(index, start, notes)
                encode(start)
                if gate is not None:
                    gate.begin_window(index)
                entries = []            # every candidate of every decode round: (hard, score, round, k, notes, parts, results, guards, muri)
                scored_rounds, round_shifts = [], []
                guard_on = scorer.expect is not None and options.guard
                n_extra_muri = options.muri_resamples if gate is not None else 0
                n_extra_guard = options.guard_rounds if guard_on else 0
                used_muri = used_guard = used_qguard = 0
                n_extra_qguard = options.qguard_rounds if (quality_on and options.qguard) else 0
                shift_now = 0.0
                rnd = 0
                Kw = chorus_state.k_for(index, K) if chorus_state is not None else K   # pool: more candidates in the chorus windows
                while True:
                    round_shifts.append(shift_now)
                    guards = [base.clone() for _ in range(Kw)]

                    def rng_factory(k, attempt, _index=index, _seed=seed, _rnd=rnd):
                        if Kw == 1 and attempt == 0 and _rnd == 0:
                            return sequential_rng            # legacy stream: one RNG across the windows
                        return np.random.default_rng([_seed, _index, k, attempt] if _rnd == 0 else
                                                     [_seed, _index, k, attempt, _rnd])

                    bias.tilt_shift = shift_now
                    results = yield from decode_candidates_co(self.model, encoded[start], prefixes,
                                                              options if plan_state is None else plan_state.window_options(options, index), self.V,
                                                              guards, rng_factory, (T_MAX if self.V.seq_version >= 6 else min(T_MAX, end_tick - start)),   # bd3: v6 always decodes the full 4-bar window like training (a truncated last window forbade BARM and forced notes into the final audio bar); HandGuard still blocks notes past end_tick
                                                              bias=bias, backend=backend,
                                                              song=w3["song"] if w3 is not None else None,
                                                              bars=((w3 or {}).get("bdi") or bdi_ids(self.V, None)) if self.V.seq_version >= 6 else None)
                    bias.tilt_shift = 0.0
                    round_notes = [canonicalize([n for n in decode_events(self.V, tokens, start) if self.V.seq_version < 6 or n.beat < end_tick]) for tokens, _, _ in results]   # bd3: drop anything past the audio end
                    assessed = gate.assess(round_notes, start, window_end) if gate is not None else [None] * Kw
                    scored = []
                    for k, ((tokens, probs, stats), new_notes) in enumerate(zip(results, round_notes)):
                        mean_lp = float(np.mean(probs)) if probs else 0.0
                        score, parts = scorer.score(new_notes, start, mean_lp, carry_end, muri=assessed[k])
                        if hands is not None and options.w_hand > 0:
                            _t = time.perf_counter()
                            parts["p_hand"] = hands.term(hands.peek(new_notes, start, window_end)[0])
                            w_hand = options.w_hand
                            if chorus_state is not None:
                                # hand: inside a chorus window the penalty is halved while the candidate is not yet at the
                                # level / audio expectation (a candidate below E cannot be "too dense")
                                h_scale = chorus_state.hand_scale(index, len(new_notes), parts.get("expected"))
                                if h_scale != 1.0:
                                    parts["hand_scale"] = float(h_scale)
                                    w_hand = options.w_hand * h_scale
                            score -= w_hand * parts["p_hand"]
                            parts["score"] = float(score)
                            q_time["hand"] += time.perf_counter() - _t
                        hard = assessed[k]["hard"] if assessed[k] is not None and assessed[k]["ok"] else 0
                        scored.append((score, k, new_notes, parts))
                        entries.append((hard, score, rnd, k, new_notes, parts, results, guards, assessed[k]))
                    scored_rounds.append(scored)
                    if gate is not None:
                        gate.rejected += sum(1 for e in entries[-Kw:] if e[0] > 0)
                    muri_ok = gate is None or any(e[0] == 0 and (e[8] is None or e[8]["penalty"] < options.muri_resample_penalty
                                                                 or not options.muri_resample_penalty) for e in entries[-Kw:])
                    guard_ok = (not guard_on) or any(e[0] == 0 and e[5].get("guard", 0) == 0 and _muri_pen(e) < GUARD_MURI_PEN
                                                     for e in entries)
                    qg = None
                    if n_extra_qguard > used_qguard:
                        # quality guard (qmode.py): every usable candidate is much sparser / denser than the level + audio expect (or leaves active music empty)
                        gs = qmode.guard_state([e[5] for e in entries if e[0] == 0])
                        if gs is not None and gs["min"] > options.qguard_resample:
                            qg = gs
                    if not muri_ok and used_muri < n_extra_muri:
                        used_muri += 1
                        shift_now = 0.0 if guard_ok else options.guard_shift
                    elif not guard_ok and used_guard < n_extra_guard:
                        used_guard += 1
                        shift_now = options.guard_shift * (1.0 if used_guard == 1 else 1.5)
                    elif qg is not None:
                        used_qguard += 1
                        shift_now = qmode.guard_shift(bias.tilt.value, qg["dir"], used_qguard)
                    else:
                        break
                    rnd += 1
                clean = [e for e in entries if e[0] == 0]
                guard_lo = None
                if quality_on and options.qguard and clean:
                    tier0 = min(_guard_tier(e) for e in clean)
                    clean = [e for e in clean if _guard_tier(e) == tier0]
                    clean, guard_lo = qmode.guard_filter(clean)
                if gate is not None:
                    gate.resampled += used_muri > 0
                    gate.fallback += not clean
                la_info = None
                if clean:
                    best = max(clean, key=lambda e: (-_guard_tier(e), e[1], -e[2], -e[3]))
                    if options.la_top > 1 and options.la_samples > 0 and index + 1 < len(starts):
                        top = qmode.pick_top(clean, options.la_top)
                        if len(top) > 1:
                            st = SimpleNamespace(index=index, start=start, window_end=window_end, notes=notes, packed=packed, scorer=scorer,
                                                 gate=gate, bias=bias, hands=hands, carry_end=carry_end)
                            _t = time.perf_counter()
                            values, la_info = yield from qmode.lookahead(q, st, top)
                            q_time["lookahead"] += time.perf_counter() - _t
                            commit = max(range(len(top)), key=lambda m: (top[m][1] + options.la_weight * values[m], -top[m][2], -top[m][3]))
                            la_info.update(chosen=commit, scores=[round(e[1], 3) for e in top], values=[round(v, 3) for v in values])
                            q_time["lookahead_windows"] += 1
                            q_time["lookahead_changed"] += int(top[commit] is not best)
                            best = top[commit]
                else:
                    best = min(entries, key=lambda e: (e[0], -e[1], e[2], e[3]))
                _, _, chosen_round, chosen, new_notes, parts, results, guards, chosen_muri = best
                scored = scored_rounds[chosen_round]
                tokens, probs, stats = results[chosen]
                scorer.accept(parts)
                if gate is not None and chosen_muri is not None and chosen_muri["ok"]:
                    gate.accept(new_notes, chosen_muri, start, window_end)
                elif gate is not None:
                    gate.accept(new_notes, dict(events=[]), start, window_end)
                notes.extend(new_notes)
                tracker, holds = guards[chosen].tracker, guards[chosen].holds
                for n in new_notes:
                    if n.kind in ("hold", "thold", "slide"):
                        carry_end = max(carry_end, int(n.end_beat()))
                previous_tokens = len(tokens)
                probabilities.extend(probs)
                stats.update(start_tick=start, notes=len(new_notes), chosen=chosen,
                             history_notes=self._history_notes(prefixes[0]["tokens"]),
                             window_candidates=Kw, touch_offset=bias.features.touch_offset,
                             chorus=bool(chorus_state is not None and chorus_state.is_chorus(index)),
                             fest_offset=bias.features.fest_offset, steer=bias.log[-1],
                             candidate_scores=[round(item[0], 4) for item in sorted(scored, key=lambda i: i[1])],
                             candidate_notes=[len(item[2]) for item in sorted(scored, key=lambda i: i[1])],
                             candidate_hand=[round(item[3].get("p_hand", 0.0), 4) for item in sorted(scored, key=lambda i: i[1])],
                             resamples_all=sum(r[2]["resamples"] for r in results),
                             fallback_all=sum(bool(r[2]["fallback"]) for r in results),
                             parts={k: v for k, v in parts.items() if not k.startswith("_")})
                stats["rounds"] = rnd + 1
                if guard_on:
                    stats["guard_rounds_used"] = used_guard
                stats["candidate_parts"] = [
                    dict(r=e[2], k=e[3], n=len(e[4]), hard=e[0], score=round(e[1], 3), lp=round(e[5]["mean_logprob"], 3),
                         E=None if e[5].get("expected") is None else round(e[5]["expected"], 1),
                         act=round(e[5]["p_activity"], 3), den=round(e[5]["p_density"], 3), int=round(e[5]["p_intensity"], 3),
                         rep=round(e[5]["p_repeat"], 3), muri=round(e[5].get("p_muri", 0.0), 2), empty=e[5]["empty_half_bars"],
                         guard=e[5].get("guard"), gd=e[5].get("guard_detail"))
                    for e in entries]
                if gate is not None:
                    stats["muri"] = dict(rounds=len(scored_rounds), chosen_round=chosen_round,
                                         counts=dict(chosen_muri["counts"]) if chosen_muri is not None else None,
                                         hard=best[0], penalty=None if chosen_muri is None else round(chosen_muri["penalty"], 3),
                                         candidate_hard=[e[0] for e in entries[-K:]])
                if self.V.seq_version >= 3:
                    stats["v3"] = dict(plan=None if w3 is None else w3["plan"], song=bool(w3 is not None and w3["song"] is not None),
                                       ref=self.V.toks[int(ref_used[0])], ref_tokens=len(ref_used[1]),
                                       ref_block=None if w3 is None or w3["ref_info"] is None else int(w3["ref_info"]["e"]),
                                       ref_sim=None if w3 is None or w3["ref_info"] is None else round(float(w3["ref_info"]["sim"]), 4),
                                       ref_shrunk=bool(w3 is not None and len(ref_used[1]) < len(w3["ref"][1])))
                    if self.V.seq_version >= 4:
                        stats["v4"] = dict(sec=self.V.toks[int(rows_w[0][self.V.sec_col - 1])],
                                           secn=self.V.toks[int(rows_w[0][self.V.secn_col - 1])])
                    if self.V.seq_version == 5:
                        stats["bd"] = [self.V.toks[int(x)] for x in rows_w[0][self.V.bd_col - 1:self.V.bd_col + 3]]
                if la_info is not None:
                    stats["lookahead"] = la_info
                if guard_lo is not None:
                    stats["guard"] = round(float(guard_lo), 4)
                if used_qguard:
                    stats["qguard_rounds"] = used_qguard
                if chosen_round and round_shifts[chosen_round]:
                    stats["tilt_shift"] = round(float(round_shifts[chosen_round]), 3)
                if hands is not None:
                    hp, hst = hands.advance(new_notes, start, window_end)
                    stats["hand"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in hst.items()}
                if quality_on or infill_on:
                    records.append(dict(index=index, start=start, end=window_end, notes=list(new_notes), tokens=list(tokens), probs=list(probs),
                                        mean_lp=float(np.mean(probs)) if probs else 0.0, scale=scorer.scale, stats=stats,
                                        rows=[list(map(int, r)) for r in rows_w],
                                        ref_block=stats.get("v3", {}).get("ref_block"), ref_tokens=stats.get("v3", {}).get("ref_tokens", 0)))
                windows.append(stats)
                log(f"candidate {candidate_index + 1}/{options.candidates} window {index + 1}/{len(starts)} "
                    f"notes={len(new_notes)} tokens={len(tokens)} retries={stats['backtracks']}"
                    + (f" pick={chosen}/{Kw}" if Kw > 1 else "")
                    + (" chorus" if stats.get("chorus") else ""), flush=True)
            refine_info, infill_info, cost_info = None, None, None
            if quality_on and options.refine_passes > 0 and records:
                pre_records = list(records)
                _t = time.perf_counter()
                refine_info = yield from qmode.self_refine(q, records, log=log)
                q_time["refine"] += time.perf_counter() - _t
                if playcheck.check(canonicalize([n for r in records for n in r["notes"]]), tempo, stats=False)["n_viol"]:
                    log("warning: self-refine produced a playcheck violation; reverting to the pre-refine chart", flush=True)
                    records[:] = pre_records
                    refine_info = dict(reverted=True, passes=refine_info)
                notes = [n for r in records for n in r["notes"]]
                windows = [r["stats"] for r in records]
                probabilities = [x for r in records for x in r["probs"]]
            if infill_on and records and getattr(q, "infill", None) is not None:
                from . import infill as infill_mod
                pre_records = list(records)
                _t = time.perf_counter()
                infill_info = yield from infill_mod.infill_refine(q, records, log=log)
                q_time["infill"] = time.perf_counter() - _t
                if playcheck.check(canonicalize([n for r in records for n in r["notes"]]), tempo, stats=False)["n_viol"]:
                    log("warning: infill refinement produced a playcheck violation; reverting to the pre-infill chart", flush=True)
                    records[:] = pre_records
                    infill_info = dict(reverted=True, passes=infill_info)
                notes = [n for r in records for n in r["notes"]]
                windows = [r["stats"] for r in records]
                probabilities = [x for r in records for x in r["probs"]]
            if quality_on and options.quality == "max" and records:
                total, comp, _ = qmode.chart_cost(q, records)
                cost_info = dict(total=round(total, 3), muri_hard=int(sum(c["hard"] for c in comp)))
            notes = canonicalize(notes)
            if bool(getattr(options, "brk_timing_gate", True)) and float(getattr(options, "brk_audio_floor", 0.0) or 0.0) > 0:
                notes = prune_breaks_by_audio(notes, mel, tempo, first, float(options.brk_audio_floor))
            pc = playcheck.check(notes, tempo)
            estimate = levelest.estimate_level(notes, tempo, slot=slot)
            mean_log_prob = float(np.mean(probabilities)) if probabilities else -1e30
            score = candidate_score(pc, estimate, target_level, mean_log_prob)
            candidates.append(dict(notes=notes, seed=seed, playcheck=pc, level_estimate=estimate,
                                   mean_log_probability=mean_log_prob, score=score, windows=windows,
                                   steer=bias.summary(), muri_gate=None if gate is None else gate.summary(),
                                   refine=refine_info, infill=infill_info, chart_cost=cost_info,
                                   hand_final=(handtravel.chart_hand_stats(notes, tempo) if quality_on else None)))
        chosen = min(range(len(candidates)), key=lambda i: candidates[i]["score"])
        if options.quality == "max" and len(candidates) > 1 and all(c.get("chart_cost") for c in candidates):
            chosen = min(range(len(candidates)), key=lambda i: (candidates[i]["playcheck"]["n_viol"], candidates[i]["chart_cost"]["total"]))
        winner = candidates[chosen]
        muri_whole = None
        if options.muri_check:
            muri_whole = murigate.whole_chart_check(winner["notes"], simai.TempoMap(list(zip(grid["beats"], grid["bpms"])), grid["first"]))
            log(f"muri check ({muri_whole.get('engine')}): "
                + (f"{muri_whole['n_events']} events, {muri_whole['per_1000_notes']}/1000 notes {muri_whole['counts']}"
                   if muri_whole.get("ok") else f"unavailable {muri_whole.get('errors')}"), flush=True)
        flags = chart_flags(winner["notes"])
        for state, enabled, name in zip((options.touch, options.ex, options.fest), flags, ("touch", "ex", "fest")):
            if state == "off" and enabled:
                raise RuntimeError(f"hard-disabled feature {name} appeared in generated notes")
        if winner["playcheck"]["n_viol"]:
            raise RuntimeError(f"hand-constrained decoding produced {winner['playcheck']['n_viol']} violations")
        info = dict(format_version=1, audio=str(audio_path), parameters=asdict(requested), auto_level=auto_level,
                    checkpoint=self.checkpoint_info, device=str(self.device), amp=self.amp, features=feat_info,
                    grid=grid, duration=duration, mel_mean=mean, mel_std=std,
                    chosen_candidate=chosen, seed=winner["seed"], slot=slot, n_notes=len(winner["notes"]),
                    empty_chart=not winner["notes"], level_estimate=winner["level_estimate"],
                    muri=dict(gate=winner.get("muri_gate"), whole_chart=muri_whole),
                    quality=dict(preset=options.quality, window_candidates=K, la_top=options.la_top, la_samples=options.la_samples,
                                 w_hand=options.w_hand, hand_mode=options.hand_mode, hand_band=handtarget.band_of(target_level),
                                 hand_kappa=options.hand_kappa, hand_z0=options.hand_z0, hand_final=winner.get("hand_final"),
                                 refine_passes=options.refine_passes, candidates=options.candidates,
                                 seconds={k: (round(v, 2) if isinstance(v, float) else v) for k, v in q_time.items()},
                                 refine=winner.get("refine"), infill=winner.get("infill"),
                                 infill_passes=options.infill_passes,
                                 chart_cost=[c.get("chart_cost") for c in candidates]),
                    playcheck={k: winner["playcheck"][k] for k in
                               ("n_viol", "n_hands_viol", "n_jump_viol", "n_events", "violations")},
                    generated_end_tick=min(end_tick, starts[-1] + T_MAX),
                    partial=starts[-1] + T_MAX < end_tick,
                    window_stats=dict(
                        windows=len(winner["windows"]),
                        backtracks=sum(w["backtracks"] for w in winner["windows"]),
                        resampled_windows=sum(w["resamples_all"] > 0 for w in winner["windows"]),
                        fallback_windows=sum(w["fallback_all"] > 0 for w in winner["windows"])),
                    candidates=[{k: v for k, v in c.items() if k not in ("notes", "playcheck")} |
                                {"n_notes": len(c["notes"]), "n_viol": c["playcheck"]["n_viol"]}
                                for c in candidates],
                    timings=dict(checkpoint_load=self.load_seconds, grid=grid_seconds, features=feature_seconds,
                                 generation=time.perf_counter() - started - grid_seconds - feature_seconds,
                                 total=time.perf_counter() - started))
        if self.V.seq_version >= 3:
            info["v3"] = v3_info
        if plan_state is not None:
            info["plan"] = plan_state.info()
        if chorus_state is not None:
            info["chorus"] = chorus_state.info()
        if self.V.seq_version >= 2:
            info.update(
                seq_version=self.V.seq_version,
                conditioning=dict(
                    names=["LV", "SLOT", "TYPE", "TOUCH", "EXN", "FEST", "DES", "VER", "DENS"] + (["PLAN"] if self.V.seq_version >= 3 else [])
                          + (["SEC", "SECN"] if self.V.seq_version >= 4 else []),
                    rows=[[self.V.toks[int(t)] for t in row] for row in rows],
                    row_layout=dict(designers=[name for _, name in designers],
                                    guidance=[name for name, _, _, _ in guidance_plan(options)],
                                    all_unk=options.cfg > 1),
                    feature_on_amount="MID",
                    level=target_level, density_token=self.V.toks[int(rows[0][8])],
                    density_source=("checkpoint.seq_meta.level_bucket/dens_edges"
                                    if target_level is not None else "auto"),
                    calibration=dict(enabled=calibration is not None,
                                     file=options.calib_file if calibration is not None else None),
                    ann=options.ann,
                ),
            )
        out = out or ROOT / "out/generated" / f"{audio_path.stem}-{options.seed}"
        write_output(out, audio_path, winner["notes"], grid, requested, info,
                     title=title or audio_path.stem, artist=artist, designers=designers)
        return info
