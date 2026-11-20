"""Muri (无理) detection wrapper around the two extracted third-party engines (third_party/muri-extracted).

Muri = chart situations that the real game's judgement makes impossible/unfair, not just a hand-count problem:
  SlideTooFast  inner-screen muri: a slide's judge areas get triggered too early / too fast
  SlideHeadTap  outer-key muri: the sliding hand hits a tap/button it passes over (jump-start of a slide head)
  TapOnSlide    tail collision: a note sits on a slide's path / end and is auto-triggered by it
  Overlap       overlapping keys (叠键)
  MultiTouch    too many simultaneous contacts (多押)

Engines
  "miacode"  MiaCode muri-cli.exe (C++/Qt6 Core, Windows only; Qt DLLs from C:\\Qt\\6.6.3\\msvc2019_64\\bin).  One process
             per call (the exe loads an 18 MB slide table every time).  Events = dynamic `diagnostics`; the user-facing
             merged panel (static+dynamic, de-duplicated) is reported as `panel_counts`.
  "dx"       MaiMuriDX (pure Python).  Runs in a persistent child process (JSON lines) so the slide tables are loaded
             once; `inproc=True` runs it inside the calling process (batch jobs).  Events = dynamic records; the
             static-checker records are reported as `static_counts`.  (dx labels slide-head hits TapOnSlide
             dynamically, MiaCode SlideHeadTap: see EQUIV.)
  "both"     both engines + `combined` summary (union / intersection by time & category).

API
  check_muri(src, slot=None, *, engine="dx", first=None, timeout=60, hand_radius=None) -> dict
      src : path of maidata.txt / its folder (needs `slot`) | (inote_text, first_seconds) | raw inote text (first=...)
      sanitize=True: if the parser rejects notes (e.g. slide shapes that do not exist in the game), strip them and retry;
      the number of stripped notes is returned as "removed_notes" (the charts are otherwise skipped with ok=False).
      drop_touch_multi=True (dx only): drop MultiTouch events that involve touch notes (MiaCode does not count touch sensors);
      engine="both" does this for dx automatically.  hand_radius: 1080-px units (MiaCode default 30, MaiMuriDX default 60).
      returns {"ok", "engine", "counts": {cat: n}, "events": [{"time","category","notes","line","col","engine",...}],
               "elapsed": sec, "error"?, per-engine extras...}; for engine="both": adds "engines" {name: result}
               and "combined" {"union": counts, "intersection": counts, "n_union", "n_inter", ...}.
  check_muri_notes(tick_notes, simai_tempomap, *, engine, ...)  same, for in-memory canonical tick notes
                                                              (written with simai.write_chart; used by generation).
  muri_events_in_range(tick_notes, tempo, t0, t1, ...)          events with t0 <= time < t1 (seconds) of notes_so_far + candidate
                                                              window; computed on a sub-chart only (~45-100 ms with engine="dx").
  check_muri_window(tick_notes, tempo, tick0, tick1, ...)      same in ticks, returns the full result dict (counts, elapsed, ...).
  filter_events(result, t0, t1)                                 time filter on a check_muri result.
  chart_inote(src, slot) -> (inote_text, first)                helper.
Times are seconds on the audio timeline (the chart `first` offset is included, as both engines do).
"""
from __future__ import annotations
import os, sys, re, json, time, atexit, subprocess, threading, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TP = ROOT / "third_party" / "muri-extracted"
DX_PKG = TP / "maimuridx"
MIA_EXE = TP / "miacode" / "bin" / "muri-cli.exe"
QT_BIN = Path(os.environ.get("MURI_QT_BIN", r"C:\Qt\6.6.3\msvc2019_64\bin"))

CATEGORIES = ("SlideTooFast", "SlideHeadTap", "TapOnSlide", "Overlap", "MultiTouch")
# categories the two engines may label differently for the same event (dx reports slide-head hits as TapOnSlide)
EQUIV = {"SlideHeadTap": "SlideCollision", "TapOnSlide": "SlideCollision"}
SHORT = {"SlideTooFast": "inner", "SlideHeadTap": "outer", "TapOnSlide": "tail", "Overlap": "overlap", "MultiTouch": "multi"}


class MuriError(RuntimeError):
    pass


# --------------------------------------------------------------------------- input
def _read_text(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return f.read().replace("\r\n", "\n")


def chart_inote(src, slot=None):
    """-> (inote_text, first_seconds).  src: maidata.txt path | folder | (text, first) | raw inote text"""
    if isinstance(src, (tuple, list)):
        return str(src[0]), float(src[1])
    if isinstance(src, str) and ("\n" in src or "," in src) and not os.path.exists(src):
        return src, 0.0          # raw inote text
    p = Path(src)
    if p.is_dir():
        p = p / "maidata.txt"
    if slot is None:
        raise ValueError("slot is required when src is a maidata file")
    raw = _read_text(p)
    fields = {}
    for m in re.finditer(r"^&([^=\n]+)=(.*?)(?=\n&|\Z)", raw, re.S | re.M):
        fields[m.group(1).strip()] = m.group(2)
    key = "inote_%d" % slot
    if key not in fields or not fields[key].strip():
        raise MuriError("no &%s in %s" % (key, p))
    fk = "first_%d" % slot if ("first_%d" % slot) in fields else "first"
    try:
        first = float(fields.get(fk, "0").strip() or 0)
    except ValueError:
        first = 0.0
    return fields[key], first


def check_input_termination(text):
    """cheap version of MaiMuriDX's end check so MiaCode gets the same inputs (generated charts always end with ',' + E)"""
    return bool(text.strip())


# --------------------------------------------------------------------------- MaiMuriDX
_dx_servers = {}
_dx_lock = threading.Lock()


class _DxProc:
    def __init__(self, config=None):
        env = dict(os.environ)
        env["MURI_DX_PKG"] = str(DX_PKG)
        env["MURI_DX_CONFIG"] = json.dumps(config or {})
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.p = subprocess.Popen([sys.executable, "-B", str(Path(__file__).with_name("_muri_dx_server.py"))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  encoding="utf-8", env=env, bufsize=1)
        hello = self.p.stdout.readline()
        if not hello or not json.loads(hello).get("ready"):
            raise MuriError("MaiMuriDX server failed to start")
        self.n = 0

    def request(self, text, first, timeout):
        self.n += 1
        req = {"id": self.n, "text": text, "first": first}
        box = []
        def _io():
            try:
                self.p.stdin.write(json.dumps(req, ensure_ascii=True) + "\n"); self.p.stdin.flush()
                box.append(self.p.stdout.readline())
            except Exception as e:      # broken pipe etc.
                box.append(None)
        th = threading.Thread(target=_io, daemon=True); th.start(); th.join(timeout)
        if th.is_alive() or not box or not box[0]:
            self.kill()
            raise MuriError("MaiMuriDX timeout/crash (%.0fs)" % timeout)
        return json.loads(box[0])

    def kill(self):
        try:
            self.p.kill()
        except Exception:
            pass

    def alive(self):
        return self.p.poll() is None


def _dx_proc(hand_radius):
    key = hand_radius
    with _dx_lock:
        pr = _dx_servers.get(key)
        if pr is None or not pr.alive():
            pr = _dx_servers[key] = _DxProc({"hand_radius_normal": hand_radius} if hand_radius is not None else None)
        return pr


@atexit.register
def _close_all():
    for pr in list(_dx_servers.values()):
        pr.kill()


_dx_inproc = {}


def _dx_run(text, first, timeout, hand_radius, inproc):
    if inproc:
        eng = _dx_inproc.get(hand_radius)
        if eng is None:
            sys.path.insert(0, str(Path(__file__).parent))
            import _muri_dx_server as S      # noqa
            sys.path.pop(0)
            eng = _dx_inproc[hand_radius] = S.DxEngine(DX_PKG, {"hand_radius_normal": hand_radius} if hand_radius is not None else None)
        return eng.run({"text": text, "first": first})
    last = None
    for attempt in range(2):
        try:
            return _dx_proc(hand_radius).request(text, first, timeout)
        except MuriError as e:
            last = e
            if "timeout" in str(e) and attempt == 0:
                break
    raise last


def _norm_dx(resp, drop_touch_multi=False):
    ev = []
    n_tm = 0
    for r in resp["dynamic"]:
        aff = r.get("affected") or {}
        notes = [aff.get("note")] if aff.get("note") else []
        cause = r.get("cause")
        line_, col_ = aff.get("line"), aff.get("col")
        if isinstance(cause, dict) and cause.get("note"):
            notes.append(cause["note"])
        elif isinstance(cause, list):             # MultiTouch: the list of simultaneously held notes
            notes += [c.get("note") for c in cause if isinstance(c, dict) and c.get("note")]
            if cause and line_ is None:
                line_, col_ = cause[0].get("line"), cause[0].get("col")
        for ja in r.get("judge_areas", []) or []:
            c = ja.get("cause")
            if isinstance(c, dict) and c.get("note") and c["note"] not in notes:
                notes.append(c["note"])
        if drop_touch_multi and r["type"] == "MultiTouch" and any(n[:1] in "ABCDE" for n in notes if n):
            n_tm += 1                 # MiaCode semantics: touch sensors do not count towards multi-touch
            continue
        ev.append({"time": float(r["time"]), "category": r["type"], "notes": notes, "line": line_,
                   "col": col_, "engine": "dx", "detail": r.get("type")})
    ev.sort(key=lambda e: e["time"])
    sc = collections.Counter(r["type"] for r in resp["static"])
    sev = []
    for r in resp["static"]:
        aff, cause = r.get("affected") or {}, r.get("cause") or {}
        sev.append({"time": None, "category": r["type"], "notes": [x for x in (aff.get("note"), cause.get("note")) if x],
                    "line": aff.get("line"), "col": aff.get("col"), "delta": r.get("delta"), "engine": "dx"})
    return ev, {"static_counts": dict(sc), "static_events": sev, "note_count": resp["note_count"], "dropped_touch_multi": n_tm}


# --------------------------------------------------------------------------- MiaCode
def _mia_env():
    env = dict(os.environ)
    if os.name == "nt" and QT_BIN.is_dir():
        env["PATH"] = str(QT_BIN) + os.pathsep + env.get("PATH", "")
    return env


def _mia_run(text, first, timeout, hand_radius):
    if os.name != "nt":
        raise MuriError("MiaCode muri-cli.exe is Windows-only")
    cmd = [str(MIA_EXE), "--headless", "--first", repr(float(first))]
    if hand_radius is not None:
        cmd += ["--hand-radius-px", str(hand_radius)]
    try:
        cp = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, env=_mia_env(), check=False)
    except subprocess.TimeoutExpired:
        raise MuriError("MiaCode timeout (%.0fs)" % timeout)
    out = cp.stdout.decode("utf-8", errors="replace")
    try:
        res = json.loads(out)
    except json.JSONDecodeError:
        raise MuriError("MiaCode returned no JSON (exit %s): %s" % (cp.returncode, cp.stderr.decode("utf-8", "replace")[:300]))
    if res.get("ok") is False or cp.returncode:
        errs = (res.get("parse") or {}).get("errors") or []
        detail = "; ".join("L%s: %s" % (e.get("line"), e.get("message")) for e in errs) or json.dumps(res, ensure_ascii=False)[:300]
        raise MuriError("MiaCode parse/run failure: " + detail)
    return res


def _norm_mia(res):
    ev = []
    for d in res.get("diagnostics", []):
        da = d.get("detail_args", {}) or {}
        notes = [da[k] for k in ("left", "right", "actions") if da.get(k)]
        ev.append({"time": float(d["second"]), "category": d["kind"], "notes": notes, "line": d.get("line"),
                   "col": d.get("col"), "engine": "miacode", "detail": d.get("detail")})
    ev.sort(key=lambda e: e["time"])
    pc = collections.Counter(p["kind"] for p in res.get("visible_panel_entries", []))
    sc = collections.Counter(p["kind"] for p in res.get("static_references", []))
    sev = []
    for r in res.get("static_references", []):
        aff, cause = r.get("affected") or {}, r.get("cause") or {}
        sev.append({"time": aff.get("second"), "category": r["kind"], "notes": [],
                    "line": aff.get("line"), "col": aff.get("col"), "delta": r.get("delta_ms"), "engine": "miacode"})
    return ev, {"panel_counts": dict(pc), "static_counts": dict(sc), "static_events": sev,
                "note_count": res.get("summary", {}).get("note_count")}


# --------------------------------------------------------------------------- public API
def _counts(events):
    c = {k: 0 for k in CATEGORIES}
    for e in events:
        c[e["category"]] = c.get(e["category"], 0) + 1
    return c


_ERR_NOTE = re.compile(r"Invalid (?:wifi slide|note): (\S+)")
_ERR_KEY = re.compile(r"KeyError: '([^']+)'")
_ERR_SHAPE = re.compile(r"unknown shape (\S+)")
_CTRL = re.compile(r"^(?:\([^)]*\)|\{[^}]*\})*")


def bad_notes_from_error(msg):
    """note strings that the parser rejected (works for both engines' messages)"""
    return _ERR_NOTE.findall(msg) + _ERR_KEY.findall(msg)


def bad_segments_from_error(msg):
    """offending slide segments ('5-4', '2z8', '8v4') named by the parser (used when a chained note cannot be located whole)"""
    return _ERR_SHAPE.findall(msg)


def strip_notes(text, bad, segs=()):
    """remove (whole) note tokens that the parser rejected.  `bad` = rejected note strings as printed by the parsers,
    `segs` = offending slide segments.  A note token is dropped when it contains a rejected string, an offending
    segment, or (for multi-arm notes 'a-b[..]*v4[..]') when start+arm equals a rejected string.
    Returns (new_text, n_removed).  An 'each' partner stays; an emptied time cell just becomes a rest."""
    n = 0
    lines = []
    for line in text.split("\n"):
        parts = re.split(r"([,/`])", line)
        for i in range(0, len(parts), 2):
            tok = parts[i]
            pre = _CTRL.match(tok).group(0)
            body = tok[len(pre):]
            if not body:
                continue
            hit = any(sg in body for sg in segs) or any(b in body for b in bad)
            if not hit and "*" in body:
                arms = body.split("*")
                recon = [arms[0]] + [body[0] + a for a in arms[1:]]
                hit = any(r == b or r.startswith(b) or b.startswith(r) for r in recon for b in bad)
            if not hit:
                for b in bad:
                    m = re.fullmatch(r"(\d)w(\d)", b)       # KeyError style key ('8w5')
                    if m and re.search(m.group(1) + r"[a-z!?$@*]*w" + m.group(2), body):
                        hit = True; break
            if hit:
                parts[i] = pre; n += 1
        lines.append("".join(parts))
    return "\n".join(lines), n


def _single(engine, text, first, timeout, hand_radius, inproc, sanitize=False, max_rounds=8, drop_touch_multi=False):
    t0 = time.perf_counter()
    out = {"ok": False, "engine": engine, "counts": {k: 0 for k in CATEGORIES}, "events": [], "first": first}
    removed = 0
    try:
        for rnd in range(max_rounds + 1):
            try:
                if engine == "dx":
                    resp = _dx_run(text, first, timeout, hand_radius, inproc)
                    if not resp.get("ok"):
                        raise MuriError(resp.get("error", "dx failure"))
                    ev, extra = _norm_dx(resp, drop_touch_multi)
                elif engine == "miacode":
                    ev, extra = _norm_mia(_mia_run(text, first, timeout, hand_radius))
                else:
                    raise ValueError("engine must be dx / miacode / both")
                out.update(ok=True, events=ev, counts=_counts(ev), **extra)
                break
            except MuriError as e:
                bad = bad_notes_from_error(str(e)) if sanitize else []
                segs = bad_segments_from_error(str(e)) if sanitize else []
                if not (bad or segs) or rnd == max_rounds:
                    raise
                text, k = strip_notes(text, bad, segs)
                if k == 0:
                    raise
                removed += k
    except MuriError as e:
        out["error"] = str(e)
    if sanitize:
        out["removed_notes"] = removed
    out["elapsed"] = time.perf_counter() - t0
    return out


def combine(a, b, tol=0.08):
    """a = miacode result, b = dx result.  Greedy match (same category, EQUIV-aware, |dt| <= tol seconds)."""
    ea, eb = a["events"], b["events"]
    used = set()
    pairs = []
    for i, x in enumerate(ea):
        best, bd = None, tol + 1e-9
        for j, y in enumerate(eb):
            if j in used:
                continue
            if y["time"] - x["time"] > tol:
                break
            cx, cy = EQUIV.get(x["category"], x["category"]), EQUIV.get(y["category"], y["category"])
            d = abs(x["time"] - y["time"])
            if cx == cy and d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best); pairs.append((i, best))
    matched_a = {i for i, _ in pairs}
    union, inter = [], []
    for i, j in pairs:
        e = dict(ea[i]); e["engine"] = "both"; e["dx_category"] = eb[j]["category"]
        union.append(e); inter.append(e)
    for i, x in enumerate(ea):
        if i not in matched_a:
            e = dict(x); e["engine"] = "miacode"; union.append(e)
    for j, y in enumerate(eb):
        if j not in used:
            e = dict(y); e["engine"] = "dx"; union.append(e)
    union.sort(key=lambda e: e["time"])
    return union, inter


def check_muri(src, slot=None, *, engine="dx", first=None, timeout=60.0, hand_radius=None, inproc=False,
               tol=0.08, sanitize=False, drop_touch_multi=False):
    """see module docstring.  Never raises for per-chart failures: returns ok=False + error."""
    try:
        text, f0 = chart_inote(src, slot)
    except (OSError, MuriError, ValueError) as e:
        return {"ok": False, "engine": engine, "error": str(e), "counts": {k: 0 for k in CATEGORIES}, "events": []}
    if first is not None:
        f0 = float(first)
    if engine in ("dx", "miacode"):
        return _single(engine, text, f0, timeout, hand_radius, inproc, sanitize, drop_touch_multi=drop_touch_multi)
    if engine != "both":
        raise ValueError("engine must be dx / miacode / both")
    t0 = time.perf_counter()
    ra = _single("miacode", text, f0, timeout, hand_radius, inproc, sanitize)
    rb = _single("dx", text, f0, timeout, hand_radius, inproc, sanitize, drop_touch_multi=True)   # "both": dx follows MiaCode semantics
    res = {"ok": ra["ok"] and rb["ok"], "engine": "both", "engines": {"miacode": ra, "dx": rb}, "first": f0}
    if res["ok"]:
        union, inter = combine(ra, rb, tol)
        res["events"] = union
        res["counts"] = _counts(union)
        res["combined"] = {"union": _counts(union), "intersection": _counts(inter), "n_union": len(union),
                           "n_inter": len(inter), "tol": tol}
    else:
        res["error"] = "; ".join(r.get("error", "") for r in (ra, rb) if not r["ok"])
        res["events"] = []; res["counts"] = {k: 0 for k in CATEGORIES}
    res["elapsed"] = time.perf_counter() - t0
    return res


def check_muri_notes(tick_notes, tempo, *, engine="dx", first=None, **kw):
    """tick_notes: canonical `mai.simai` tick Notes; tempo: `mai.simai.TempoMap` (has .first).  Written back to simai text
    with simai.write_chart (the same round-trip the generator output goes through)."""
    from .simai import write_chart
    text = write_chart(tick_notes, tempo)
    f0 = float(getattr(tempo, "first", 0.0)) if first is None else float(first)
    return check_muri((text, f0), engine=engine, **kw)


def check_muri_window(tick_notes, tempo, tick0, tick1, *, pre_sec=4.0, post_sec=3.0, engine="dx", categories=None, **kw):
    """Incremental check: events in the tick range [tick0, tick1) of a chart given as canonical tick notes
    (`notes_so_far + candidate window`), computed on a sub-chart that only contains the notes whose [start, end] time span
    touches [t0 - pre_sec, t1 + post_sec].  Muri interactions are local (a slide's duration + judge tolerances), so
    events inside the window equal the full-chart ones as long as pre_sec covers the longest slide/hold still running at
    tick0 (verified in docs/status/muri.md).  Returns {"events": [...window events...], "elapsed", "n_sub_notes", ...}.
    tempo: `mai.simai.TempoMap` (with .first)."""
    from .simai import write_chart, TPB
    f0 = float(getattr(tempo, "first", 0.0))
    t0 = f0 + tempo.to_sec(tick0 / TPB)
    t1 = f0 + tempo.to_sec(tick1 / TPB)
    sel = []
    for n in tick_notes:
        ts = f0 + tempo.to_sec(n.beat / TPB)
        te = f0 + tempo.to_sec(n.end_beat() / TPB)
        if te >= t0 - pre_sec and ts <= t1 + post_sec:
            sel.append(n)
    t_start = time.perf_counter()
    text = write_chart(sel, tempo)
    res = check_muri((text, f0), engine=engine, **kw)
    res["events"] = filter_events(res, t0, t1, categories)
    res["counts"] = _counts(res["events"])
    res["n_sub_notes"] = len(sel)
    res["elapsed_total"] = time.perf_counter() - t_start
    return res


def filter_events(result, t0, t1, categories=None):
    """events of a check_muri result with t0 <= time < t1 (seconds, audio timeline)."""
    ev = [e for e in result["events"] if t0 <= e["time"] < t1]
    if categories is not None:
        ev = [e for e in ev if e["category"] in categories]
    return ev


def muri_events_in_range(tick_notes, tempo, t0, t1, *, engine="dx", categories=None, **kw):
    """The API proposed for the per-window reranker: muri events (list of dicts: time, category, notes, line, col, engine)
    whose time lies in [t0, t1) seconds (audio timeline, `first` included), for the chart `notes_so_far + candidate window`
    given as canonical tick notes + `mai.simai.TempoMap`.  Computed on a sub-chart (see check_muri_window); extra keyword
    arguments (pre_sec, post_sec, sanitize, timeout, hand_radius, inproc) are passed through."""
    from .simai import TPB
    f0 = float(getattr(tempo, "first", 0.0))
    k0 = int(round(tempo.to_beat(t0 - f0) * TPB))
    k1 = int(round(tempo.to_beat(t1 - f0) * TPB))
    r = check_muri_window(tick_notes, tempo, k0, k1, engine=engine, categories=categories, **kw)
    if not r["ok"]:
        raise MuriError(r.get("error", "muri window check failed"))
    return r["events"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="run muri detection on one chart")
    ap.add_argument("path"); ap.add_argument("--slot", type=int); ap.add_argument("--engine", default="dx")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    r = check_muri(a.path, a.slot, engine=a.engine, timeout=a.timeout)
    print(json.dumps({k: v for k, v in r.items() if k != "engines"}, ensure_ascii=False, indent=1)[:4000])
