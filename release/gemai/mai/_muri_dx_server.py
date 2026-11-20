"""MaiMuriDX engine kept alive between charts; used by mai.muri (as a child process or in-process).

Mirrors third_party/muri-extracted/maimuridx/_worker.py::run() (same parse -> static -> dynamic order, same
parse-failure test) but keeps the upstream modules loaded between requests, so a chart does not pay interpreter
start-up + slide-table init each time.  Read-only w.r.t. third_party/: core.py is copied to a temp dir together
with the config (exactly what _worker.py does).  The config is fixed per engine (partial override of upstream
config.json, e.g. {"hand_radius_normal": 30}; for the child process: env MURI_DX_CONFIG as JSON).

Child-process protocol (JSON lines):
  request : {"id": n, "text": inote, "first": sec, "max_ticks": optional int}
  response: {"id": n, "ok": true, "static": [...], "dynamic": [...], "note_count": n, "action_count": n, "dt": sec}
            {"id": n, "ok": false, "error": "..."}
"""
import sys, os, io, json, re, math, shutil, tempfile, time, importlib.util, traceback
from pathlib import Path


class _Null:
    def write(self, s): return len(s)
    def flush(self): pass


class DxEngine:
    def __init__(self, pkg, config=None):
        self.pkg = Path(pkg).resolve()
        up = self.pkg / "upstream"
        spec = importlib.util.spec_from_file_location("_muri_dx_worker_mod", str(self.pkg / "_worker.py"))
        self.wk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.wk)
        cfg = json.loads((up / "config.json").read_text(encoding="utf-8"))
        for k, v in (config or {}).items():
            if k not in cfg:
                raise ValueError("Unknown config key: %s" % k)
            cfg[k] = v
        self.tmp = tempfile.mkdtemp(prefix="maimuridx-srv-")
        shutil.copyfile(up / "core.py", Path(self.tmp) / "core.py")
        (Path(self.tmp) / "config.json").write_text(json.dumps(cfg, ensure_ascii=True), encoding="utf-8")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(up))
        sys.path.insert(0, self.tmp)
        old = sys.stdout
        sys.stdout = _Null()          # upstream print()s its report lines
        try:
            from core import REPORT_WRITER, JUDGE_TPS
            from judge import JudgeManager, StaticMuriChecker
            from majparse import NoteActionConverter, SimaiParser
            from slides import init as init_slides
            init_slides()
        finally:
            sys.stdout = old
        self.TPS = JUDGE_TPS
        self.RW, self.JM, self.SC, self.NAC, self.SP = REPORT_WRITER, JudgeManager, StaticMuriChecker, NoteActionConverter, SimaiParser

    def run(self, req):
        rid = req.get("id")
        t0 = time.perf_counter()
        old = sys.stdout
        sys.stdout = _Null()
        try:
            text = req["text"]
            self.wk.check_input_termination(text)
            self.RW.buf = io.StringIO()
            chart = self.SP.parse_simai_chart(text, float(req.get("first", 0.0)))
            diag = self.RW.buf.getvalue()
            failures = [l for l in diag.splitlines() if re.search(r"\b(?:invalid|unknown|error)\b", l, re.I)]
            if failures:
                raise ValueError("Simai parse error: " + "\n".join(failures[:40]))
            actions = self.NAC.generate_action(chart)
            total = len(chart)
            static = self.SC.check(chart)
            manager = self.JM()
            manager.load_chart(chart, actions)
            tmax = int(req.get("max_ticks", 0)) or None
            skip = req.get("idle_skip", True)
            n = 0
            while manager.note_pointer < total or manager.active_notes:
                if skip and not manager.active_notes and not manager.active_actions and manager.pad_states == 0:
                    # idle: tick() would only advance the clock.  Jump to just before the next activation
                    # (notes activate at moment - 2*TPS, actions at moment - TPS).  Bit-exact w.r.t. 1-tick stepping.
                    thr = []
                    if manager.note_pointer < total:
                        thr.append(chart[manager.note_pointer].moment - 2 * self.TPS)
                    if manager.action_pointer < len(actions):
                        thr.append(actions[manager.action_pointer].moment - self.TPS)
                    if thr:
                        tgt = math.ceil(min(thr)) - 1
                        if tgt > manager.timer:
                            manager.timer = tgt
                manager.tick(1)
                n += 1
                if tmax and n > tmax:
                    raise RuntimeError("tick budget exceeded")
            resp = {"id": rid, "ok": True, "static": static, "dynamic": manager.muri_record_list,
                    "note_count": total, "action_count": len(actions), "dt": time.perf_counter() - t0}
        except BaseException as exc:      # keep serving
            resp = {"id": rid, "ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "dt": time.perf_counter() - t0}
        finally:
            sys.stdout = old
            self.RW.buf = io.StringIO()
        return resp

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def serve():
    proto = sys.stdout
    sys.stdout = _Null()
    eng = DxEngine(os.environ["MURI_DX_PKG"], json.loads(os.environ.get("MURI_DX_CONFIG", "{}")))
    proto.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n"); proto.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp = eng.run(json.loads(line))
        proto.write(json.dumps(resp, ensure_ascii=True, allow_nan=False, default=str) + "\n"); proto.flush()
    eng.close()


if __name__ == "__main__":
    try:
        serve()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
