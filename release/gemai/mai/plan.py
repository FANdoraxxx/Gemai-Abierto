"""Whole-song planning layer for the generator (scratch/plan: section predictor + planner), behind ``--plan S`` (default 0 = off).

  plan_state = build(gen, v3ctx, level, options, nw, log)      # once per song (CPU, ~50 ms); None = off / not possible
  plan_state.expect(curve_ratios_model)                        # -> (plan_ratios, EXPECT params) for activity.WindowExpectation
  plan_state.apply_tokens(v3ctx, stats)                        # PLAN token of every window <- planned density bucket
  Steer(..., plan=plan_state)                                  # per-window share targets + feed-forward biases + creativity

S is the strength of the plan: every planned quantity enters as ``value ** S`` (S = 1: the planner's full contrast, S = 0: nothing).
The planner (scratch/plan/planner.py) paired-contrast tables are fitted on official MiLK+ charts only; see scratch/plan/INTEGRATION.md.
``--plan-parts`` selects which parts act (default dens,token,tilt,rhythm):
  dens   window densities of the expectation (rerank / guards) and of the Steer density feedback
  token  PLAN condition token <- planned density bucket
  tilt   feed-forward density slope on the window transition (ff x (E_w - E_mean) / 16 beats)
  rhythm per-window 16th/24th share targets, feed-forward on non-8th time tokens and on continuing a 16th stream
  wide   feed-forward on >= 3-lane jumps within an 8th
  share  per-window share targets of hold / slide / BRK / EX / double / touch / connected slides (the generator is already above the
         official chorus/verse contrast for these on e25: leave off unless the base model changes)
  ff     feed-forward logit offsets on those kind shares
  creat  per-window content temperature / anti-repeat / motif
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from . import paths as _paths

PROJECT = _paths.project_root(__file__)
# frozen release: MAI_PLAN_DIR points at the snapshot's own copy of scratch/plan (unset = project tree, unchanged)
PLAN_DIR = Path(os.environ["MAI_PLAN_DIR"]) if os.environ.get("MAI_PLAN_DIR") else PROJECT / "scratch" / "plan"
SECPRED = PROJECT / os.environ.get("MAI_SECTIONPRED", "out/plan/sectionpred.pkl")
ALL_PARTS = ("dens", "token", "tilt", "rhythm", "wide", "share", "ff", "creat")
DEFAULT_PARTS = "dens,token,tilt,rhythm"
_MODS = None
_REC_CACHE = {}


def parse_parts(text):
    parts = {p.strip() for p in str(text).split(",") if p.strip()}
    bad = parts - set(ALL_PARTS)
    if bad:
        raise ValueError(f"plan_parts: unknown {sorted(bad)}; choose from {ALL_PARTS}")
    return parts


class _plan_imports:
    """scratch/plan has its own train.py (section-predictor helpers) that sectionpred imports lazily by name: while a planner call runs, put
    scratch/plan first on sys.path and its train module in sys.modules['train'], then restore both (the generator tree has other `train` modules)."""

    def __enter__(self):
        self.path = list(sys.path)
        self.mod = sys.modules.get("train")
        sys.path.insert(0, str(PLAN_DIR))
        if _PLAN_TRAIN is not None:
            sys.modules["train"] = _PLAN_TRAIN
        return self

    def __exit__(self, *exc):
        sys.path[:] = self.path
        if self.mod is None:
            sys.modules.pop("train", None)
        else:
            sys.modules["train"] = self.mod
        return False


_PLAN_TRAIN = None


def _mods():
    """(sectionpred, planner) from scratch/plan"""
    global _MODS, _PLAN_TRAIN
    if _MODS is None:
        import importlib.util
        with _plan_imports():
            spec = importlib.util.spec_from_file_location("train", str(PLAN_DIR / "train.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["train"] = mod
            spec.loader.exec_module(mod)
            _PLAN_TRAIN = mod
            import sectionpred as SP
            import planner as PL
            SP.load(str(SECPRED))
        _MODS = (SP, PL)
    return _MODS


def predicted_record(tab, nb=None):
    """section record (nb, starts, labels, mbar, mkind) predicted from the bar table; cached per table."""
    tab = np.asarray(tab)
    nb = int(nb or len(tab))
    key = (tab.shape, nb, float(np.asarray(tab[:, :64], np.float64).sum()))
    if key not in _REC_CACHE:
        SP, _ = _mods()
        if len(_REC_CACHE) > 8:
            _REC_CACHE.clear()
        with _plan_imports():
            _REC_CACHE[key] = SP.predict_from_table(np.asarray(tab, np.float32), nb)
    return _REC_CACHE[key]


def bucket_of(gen, ratio):
    """PLAN bucket of a window / song-mean density ratio, on the same (rank) scale the model's own PLAN predictions use"""
    from . import v3 as V3
    q = gen.v3_stats.get("plan_q")
    r = np.asarray(ratio, np.float64)
    if q is None:
        return np.clip(np.searchsorted(np.asarray(gen.v3_stats["plan_edges"], np.float64), r), 0, V3.N_PLAN - 1)
    q = np.asarray(q, np.float64)
    f = np.interp(r, q, np.linspace(0.0, 1.0, len(q)))
    return np.clip((f * V3.N_PLAN).astype(int), 0, V3.N_PLAN - 1)


# share metric of the planner (cmet name) -> what the Steer does with it
MULT_FOR = {"hold": "hold_share", "slide": "slide_share", "brk": "brk_rate", "ex": "ex_rate", "double": "each_rate",
            "tch": "touch_share", "cont": "conn_slide", "fine": "rhythm_fine", "stream": "stream", "wide": "wide"}


class PlanState:
    def __init__(self, plan, strength, parts, ff, level, source, nw):
        _, PL = _mods()
        self.plan, self.S, self.parts, self.ff, self.level, self.source = plan, float(strength), set(parts), float(ff), float(level), source
        self.arr = PL.window_arrays(plan)
        self.nw = int(nw)
        self.n_plan = len(self.arr["rel_density"])
        self.summary = plan["summary"]
        self.song = plan["song"]

    # ------------------------------------------------------------------ lookups (window index = bar // 4)
    def _get(self, key, i, default):
        a = self.arr.get(key)
        if a is None or not (0 <= i < len(a)):
            return default
        v = float(a[i])
        return default if not math.isfinite(v) else v

    def rel(self, i):
        return max(self._get("rel_density", i, 1.0), 0.05)

    def label(self, i):
        a = self.arr["label"]
        return str(a[i]) if 0 <= i < len(a) else ""

    def mult(self, name, i):
        """planned multiplier of a share, strength applied: m ** S (clipped to [0.25, 4])"""
        m = self._get("mult_" + MULT_FOR[name], i, 1.0)
        return float(np.clip(max(m, 1e-3) ** self.S, 0.25, 4.0))

    def log_ff(self, name, i):
        return self.ff * math.log(self.mult(name, i))

    def creativity(self, i):
        out = {}
        for k in ("content_temp", "anti_repeat", "motif"):
            v = self._get(k, i, 1.0)
            out[k] = float(max(v, 1e-3) ** self.S)
        return out

    # ------------------------------------------------------------------ generator-side plumbing
    def density_inputs(self, gen, v3ctx, starts, model_ratios):
        """plan_ratios + EXPECT params for activity.WindowExpectation.  model_ratios = the generator's own PLAN ratios (or None)."""
        n = len(starts)
        S = self.S
        wm = max(0.0, 1.0 - S)
        rel = np.array([self.rel(i) for i in range(n)])
        mod = np.ones(n) if model_ratios is None else np.maximum(np.asarray(model_ratios, np.float64), 0.05)
        lr = 0.45 * wm * np.log(mod) + S * np.log(rel)
        from . import activity
        params = dict(BETA_A=activity.EXPECT_PARAMS["BETA_A"] * wm, BETA_P=1.0)
        return np.exp(lr), params

    def apply_tokens(self, gen, v3ctx, starts, model_ratios):
        """overwrite the song's per-bar PLAN buckets by the planned (and strength-blended) window densities"""
        n = len(starts)
        S = self.S
        wm = max(0.0, 1.0 - S)
        rel = np.array([self.rel(i) for i in range(n)])
        mod = np.ones(n) if model_ratios is None else np.maximum(np.asarray(model_ratios, np.float64), 0.05)
        lr = wm * np.log(mod) + S * np.log(rel)
        buckets = bucket_of(gen, np.exp(lr))
        plans = np.array(v3ctx.plans, copy=True)
        for i in range(n):
            plans[i * 4:(i + 1) * 4] = int(buckets[i])
        v3ctx.plans = plans
        return [int(b) for b in buckets]

    def expected_fn(self, sc, starts):
        """expected notes per window for the Steer's density feedback (replaces the flux-table expectation while ``dens`` is on):
        log E = (1-S) log table + S log (plan expectation x k), k = sum(table) / sum(plan expectation) so the song total stays the
        level-calibrated one and only the distribution over the windows follows the plan."""
        table = np.array([sc.expected_notes_table(st) or 0.0 for st in starts], np.float64)
        plan = np.array([sc.expect.window(st) or 0.0 for st in starts], np.float64)
        ok = (table > 0.5) & (plan > 0.5)
        k = float(table[ok].sum() / plan[ok].sum()) if ok.any() else 1.0
        S = self.S
        out = table.copy()
        lr = np.zeros(len(starts))
        lr[ok] = np.clip(np.log(plan[ok] * k / table[ok]), -1.2, 1.2)
        out[ok] = table[ok] * np.exp(S * lr[ok])
        self.expected_windows = out
        self.k_expect = k

        def fn(tick, _o=out, _t=sc):
            w = int(tick) // 768
            return float(_o[w]) if 0 <= w < len(_o) else (_t.expected_notes_table(tick) or 0.0)
        return fn

    def window_options(self, options, i):
        """options with the planned content temperature of window i (creat part), else the same object"""
        if "creat" not in self.parts:
            return options
        from dataclasses import replace
        c = self.creativity(i)["content_temp"]
        if abs(c - 1.0) < 1e-6:
            return options
        return replace(options, content_temperature=options.content_temperature * c)

    def info(self):
        ws = self.plan["windows"]
        return dict(strength=self.S, parts=sorted(self.parts), ff=self.ff, level=self.level, source=self.source,
                    summary=self.summary, labels=[w["label"] for w in ws],
                    rel_density=[w["rel_density"] for w in ws], n_windows=len(ws))


def build(gen, v3ctx, level, options, nw, log=print):
    """PlanState for this song or None (plan off / no level / no bar table / grid mismatch)"""
    S = float(options.plan)
    if S <= 0:
        return None
    if v3ctx is None:
        log("warning: --plan needs the seq_version 3+ song context (MERT features); plan switched off", flush=True)
        return None
    if level is None:
        log("warning: --plan needs a level (--level N or auto); plan switched off", flush=True)
        return None
    _, PL = _mods()
    tab = np.asarray(v3ctx.tab)
    nb = len(tab)
    parts = parse_parts(options.plan_parts)
    if options.plan_file:
        plan = PL.load_plan(options.plan_file)
        if int(plan["song"]["nb"]) != nb:
            log(f"warning: plan file nb {plan['song']['nb']} != generation grid {nb}; plan switched off", flush=True)
            return None
        source = "file:" + str(options.plan_file)
    else:
        rec = predicted_record(tab, nb)
        with _plan_imports():
            plan = PL.plan(rec, np.asarray(tab, np.float32), float(level), ab=PL.fit_level_density())
        source = "pred"
    st = PlanState(plan, S, parts, options.plan_ff, level, source, nw)
    log(f"plan: strength {S:g} parts {','.join(sorted(parts))} source {source}, {len(plan['windows'])} windows, "
        f"planned C/V npb {plan['summary'].get('target_CV')}", flush=True)
    return st
