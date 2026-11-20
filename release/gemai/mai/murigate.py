"""Window-level muri (无理) gate for the generation reranker (docs/MURI_PLAN.md section 3).

For every candidate of a generation window the muri events inside the new window are computed with the windowed
MaiMuriDX engine (persistent worker, ``mai.muri``; ``drop_touch_multi`` = MiaCode semantics: touch chords do not count as
MultiTouch).  Categories (dx labels slide-head hits TapOnSlide, MiaCode SlideHeadTap: tail and outer-key events are one
category here):

* hard    Overlap always; TapOnSlide / SlideHeadTap (tail collision, outer key) when the requested or predicted level is
          below ``hard_level`` (14): a candidate with a hard event is rejected; when every candidate of a round has one
          the window is resampled (bounded) and finally the least-bad candidate is taken.
* soft    SlideTooFast (inner screen), MultiTouch (non-touch) always, TapOnSlide / SlideHeadTap from ``hard_level`` on:
          penalty per event, cheap while the chart is inside the official per-level budget (events per chart, official
          Master / Re:MASTER, docs/status/muri.md) and full price beyond it.

Events of the history that is already in the chart are common to all candidates; they are removed ("baseline") when every
candidate of a round has a hard event, so that unavoidable events do not trigger pointless resampling.
"""
from __future__ import annotations

from collections import Counter
import time

from . import muri
from .simai import TPB, canon_key, write_chart
from .tokens import note_sort_key

# official Master / Re:MASTER dynamic events per chart by display level (MiaCode; docs/status/muri.md section 3)
OFFICIAL_BUDGET = {
    "<=11": dict(inner=0.026, outer=0.0, tail=0.0, multi=0.0),
    "12": dict(inner=0.142, outer=0.062, tail=0.016, multi=0.043),
    "13": dict(inner=0.157, outer=0.079, tail=0.048, multi=0.050),
    "13+": dict(inner=0.153, outer=0.056, tail=0.035, multi=0.086),
    "14": dict(inner=0.136, outer=0.178, tail=0.216, multi=0.047),
    "14+": dict(inner=0.011, outer=0.455, tail=0.170, multi=0.375),
}
CAT_KEY = {"SlideTooFast": "inner", "SlideHeadTap": "outer", "TapOnSlide": "tail", "Overlap": "overlap", "MultiTouch": "multi"}
SOFT_WEIGHT = dict(inner=1.0, outer=1.0, tail=1.0, multi=1.5)


def level_key(level):
    if level is None:
        return "13"
    base = int(level + 1e-6)
    plus = level - base + 1e-6 >= 0.6
    if base <= 11:
        return "<=11"
    if base >= 14:
        return "14+" if (plus or base >= 15) else "14"
    return f"{base}+" if plus and base == 13 else str(base)


def static_overlaps(notes, start, end):
    """Overlap (叠键) pairs the dynamic engine can miss: a note that starts on a button / sensor while (or in the tick the hold ends) a hold or touch-hold on
    the same pad is still running.  Counts notes starting in [start, end) ticks; ``notes`` = history + candidate."""
    holds = {}                                   # pad -> end tick of the running hold (tap-like pads are keyed by kind class)
    n = 0
    for note in sorted(notes, key=lambda x: (x.beat, 0 if x.kind in ("hold", "thold") else 1)):
        pad = ("s", note.pos) if note.kind in ("touch", "thold") else ("b", note.pos)
        busy = holds.get(pad, 0)
        if note.beat <= busy and busy > 0 and start <= note.beat < end:
            n += 1
        if note.kind in ("hold", "thold"):
            holds[pad] = max(busy, note.beat + note.dur)
    return n


class MuriGate:
    def __init__(self, tempo, level, *, w_hard=5.0, w_soft=1.0, hard_level=14.0, slack=0.1, n_windows=1, pre_sec=2.0, post_sec=1.5,
                 margin_sec=0.5, hand_radius=30, timeout=20.0, log=print):
        """tempo: ``mai.simai.TempoMap`` of the chart (with ``.first``)."""
        self.tm = tempo
        self.first = float(getattr(tempo, "first", 0.0))
        self.level = level
        self.key = level_key(level)
        self.budget = OFFICIAL_BUDGET.get(self.key, OFFICIAL_BUDGET["14+"])
        self.hard_tail = level is None or level < hard_level
        self.w_hard, self.w_soft = float(w_hard), float(w_soft)
        self.slack = float(slack)
        self.n_windows = max(1, int(n_windows))
        self.pre, self.post, self.margin = pre_sec, post_sec, margin_sec
        self.hand_radius, self.timeout = hand_radius, timeout
        self.log = log
        self.hist = []                       # (t_start, t_end, note) of the accepted chart
        self.used = Counter()                # accepted events per category key
        self.calls = 0
        self.seconds = 0.0
        self.errors = 0
        self.rejected = 0                    # candidates with a hard event
        self.resampled = 0                   # windows that needed a resample round
        self.fallback = 0                    # windows taken with a hard event left

    # ------------------------------------------------------------------ engine
    def _sec(self, tick):
        return self.first + self.tm.to_sec(tick / TPB)

    def _events(self, notes, t0, t1):
        """dx events with t0 <= time < t1 for the sub-chart ``notes`` or None when the engine failed."""
        text = write_chart(sorted(notes, key=note_sort_key), self.tm)
        started = time.perf_counter()
        res = muri.check_muri((text, self.first), engine="dx", timeout=self.timeout, hand_radius=self.hand_radius,
                              drop_touch_multi=True)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        if not res["ok"]:
            self.errors += 1
            return None
        return [e for e in res["events"] if t0 <= e["time"] < t1]

    def _select(self, notes_with_times, t0, t1):
        lo, hi = t0 - self.pre, t1 + self.post
        return [n for ts, te, n in notes_with_times if te >= lo and ts <= hi]

    def _timed(self, notes):
        return [(self._sec(n.beat), self._sec(n.end_beat()), n) for n in notes]

    # ------------------------------------------------------------------ scoring
    def assess(self, candidates, start, end):
        """candidates: list of new-note lists (canonical tick notes).  Returns one dict per candidate:
        ``counts`` (category key -> events in [window start, window end + margin)), ``hard`` (int), ``penalty`` (float),
        ``ok`` (False when the engine failed for it)."""
        t0, t1 = self._sec(start), self._sec(end)
        hist = self._select(self.hist, t0, t1)
        out, seen = [], {}
        for notes in candidates:
            key = tuple(canon_key(n) for n in notes)
            if key not in seen:                      # identical candidates (low-entropy windows) share one engine call
                if notes:
                    cand = self._timed(notes)
                    summary = self._summary(self._events(hist + [n for _, _, n in cand], t0, t1 + self.margin))
                    extra = static_overlaps(hist + notes, start, end)
                    if extra and summary["ok"] and not summary["counts"].get("overlap", 0):
                        summary["counts"]["overlap"] = extra
                        summary.update(self._score(summary["counts"]))
                    seen[key] = summary
                else:
                    seen[key] = self._summary([])
            out.append(dict(seen[key]))
        if out and all(o["ok"] and o["hard"] > 0 for o in out):
            base = self._events(hist, t0, t1 + self.margin)             # events the history alone already causes
            if base is not None:
                bc = Counter(CAT_KEY[e["category"]] for e in base)
                for o in out:
                    o["counts"] = Counter({k: max(0, v - bc.get(k, 0)) for k, v in o["counts"].items()})
                    o.update(self._score(o["counts"]))
        for o in out:
            o["t1"] = t1
        return out

    def _summary(self, events):
        if events is None:
            return dict(ok=False, counts=Counter(), hard=0, penalty=0.0, events=[])
        counts = Counter(CAT_KEY[e["category"]] for e in events)
        return dict(ok=True, counts=counts, events=events, **self._score(counts))

    def _score(self, counts):
        hard = counts.get("overlap", 0) + (counts.get("outer", 0) + counts.get("tail", 0) if self.hard_tail else 0)
        penalty = self.w_hard * (hard + 1.0 * counts.get("overlap", 0))     # Overlap counts double
        soft = ["inner", "multi"] + ([] if self.hard_tail else ["outer", "tail"])
        for cat in soft:
            n = counts.get(cat, 0)
            if not n:
                continue
            free = max(0.0, self.allowance(cat) - self.used[cat])           # events still inside the official budget
            inside = min(n, free)
            penalty += self.w_soft * SOFT_WEIGHT[cat] * (0.25 * inside + (n - inside))
        return dict(hard=hard, penalty=float(penalty))

    def allowance(self, cat):
        """Events the chart may have accumulated by the end of the current window: official mean per chart x progress, plus
        half an event of slack (muri in official charts is a rare, clustered gimmick)."""
        return self.budget.get(cat, 0.0) * self._progress + self.slack

    _progress = 0.0

    def begin_window(self, index):
        self._progress = (index + 1) / self.n_windows

    # ------------------------------------------------------------------ state
    def accept(self, chosen_notes, assessed, start, end):
        """Fold the chosen candidate into the history.  Only events strictly inside the window are charged to the budget
        (events in the margin after it are recounted by the next window)."""
        self.hist.extend(self._timed(chosen_notes))
        t1 = assessed.get("t1", self._sec(end)) - self.margin
        for e in assessed.get("events", ()):
            if e["time"] < t1:
                self.used[CAT_KEY[e["category"]]] += 1

    def summary(self):
        return dict(level_key=self.key, hard_tail=self.hard_tail, used=dict(self.used), calls=self.calls,
                    seconds=round(self.seconds, 2), errors=self.errors, rejected_candidates=self.rejected,
                    resampled_windows=self.resampled, fallback_windows=self.fallback,
                    budget={k: round(self.allowance(k), 2) for k in ("inner", "outer", "tail", "multi")})


def whole_chart_check(notes, tempo, *, engine="miacode", max_events=60):
    """MiaCode (Windows) check of the final chart, falling back to the dx engine (drop_touch_multi, hand radius 30).
    Returns a json-able dict for gen_info.json: engine, ok, counts per MiaCode category, n_events, events per 1000 notes."""
    started = time.perf_counter()
    res = None
    errors = []
    for eng in ((engine, "dx") if engine != "dx" else ("dx",)):
        kw = dict(drop_touch_multi=True, hand_radius=30) if eng == "dx" else {}
        try:
            res = muri.check_muri_notes(notes, tempo, engine=eng, timeout=120.0, **kw)
        except Exception as exc:          # never let the check break a generation
            res = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        if res.get("ok"):
            break
        errors.append(f"{eng}: {res.get('error')}")
    n = len(notes)
    if not res or not res.get("ok"):
        return dict(ok=False, engine=engine, errors=errors, seconds=round(time.perf_counter() - started, 2))
    ev = res["events"]
    out = dict(ok=True, engine=res["engine"], counts={k: v for k, v in res["counts"].items()}, n_events=len(ev),
               n_notes=n, per_1000_notes=round(1000.0 * len(ev) / max(1, n), 3),
               events=[[round(e["time"], 2), e["category"]] for e in ev[:max_events]],
               seconds=round(time.perf_counter() - started, 2))
    if res.get("panel_counts") is not None:
        out["panel_counts"] = res["panel_counts"]
    if errors:
        out["fallback_errors"] = errors
    return out
