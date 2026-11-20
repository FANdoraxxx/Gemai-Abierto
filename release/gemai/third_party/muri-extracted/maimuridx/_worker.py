"""Private one-shot worker. Do not import upstream modules in the parent."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback

UPSTREAM_COMMIT = "0ebab7ff6853a159ef6af76d9e3c43b23157bb5e"
UPSTREAM = Path(__file__).resolve().parent / "upstream"


def check_input_termination(text: str) -> None:
    """Detect input that upstream would silently leave in its token buffers.

    This follows only the original top-level lexical controls. It does not
    rewrite note syntax or attempt to replace the original Simai parser.
    A standalone final E is the conventional chart-end marker; E1..E8 remain
    note tokens and therefore require a normal delimiter.
    """
    if not text.strip():
        raise ValueError("Simai parse error: input is empty")
    current_note = ""
    each_pending = False
    saw_content = False
    pending_line = pending_col = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        chars = enumerate(line, 1)
        for column, ch in chars:
            if ch == "|" and column < len(line) and line[column] == "|":
                break
            if ch.isspace():
                continue
            saw_content = True
            # The upstream HS recognizer accepts any content after HS*.
            is_hs = ch == "H" and line[column:column + 2] == "S*"
            if ch in "({" or is_hs:
                if current_note:
                    raise ValueError(
                        f"Simai parse error: note {current_note!r} at L{pending_line},C{pending_col} "
                        "has no delimiter before a control command"
                    )
                closing = ")" if ch == "(" else "}" if ch == "{" else ">"
                for _, control in chars:
                    if control == closing:
                        break
                else:
                    raise ValueError(f"Simai parse error: unclosed control at L{lineno},C{column}")
                continue
            if ch in ",/`":
                if current_note:
                    each_pending = ch == "/"
                current_note = ""
                continue
            if not current_note and ch in "12345678ABCDE":
                pending_line, pending_col = lineno, column
                current_note = ch
            elif current_note:
                current_note += ch
    if not saw_content:
        raise ValueError("Simai parse error: input contains only whitespace or comments")
    if current_note and current_note != "E":
        raise ValueError(
            f"Simai parse error: unterminated note {current_note!r} at L{pending_line},C{pending_col}; "
            "add a comma before EOF or the final E marker"
        )
    if each_pending:
        raise ValueError("Simai parse error: unterminated each group; finish the group with a note followed by a comma")


def run(request: dict) -> dict:
    check_input_termination(request["text"])
    config = json.loads((UPSTREAM / "config.json").read_text(encoding="utf-8"))
    for key, value in request["config"].items():
        if key not in config:
            raise ValueError(f"Unknown config key: {key}")
        if isinstance(config[key], bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a JSON boolean")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a nonnegative finite number")
        config[key] = value

    # core.py resolves config beside its own __file__ and generates it if absent.
    # Copy this one unchanged source file into a temporary directory with an
    # explicit config, so neither source files nor user config are ever written.
    with tempfile.TemporaryDirectory(prefix="maimuridx-") as temporary:
        temporary = Path(temporary)
        shutil.copyfile(UPSTREAM / "core.py", temporary / "core.py")
        (temporary / "config.json").write_text(json.dumps(config, ensure_ascii=True), encoding="utf-8")
        sys.path.insert(0, str(UPSTREAM))
        sys.path.insert(0, str(temporary))
        sys.dont_write_bytecode = True
        chatter = io.StringIO()
        with redirect_stdout(chatter):
            from core import JUDGE_TPS, REPORT_WRITER
            from judge import JudgeManager, StaticMuriChecker
            from majparse import NoteActionConverter, SimaiParser
            from slides import init as init_slides

            init_slides()
            parse_start = REPORT_WRITER.buf.tell()
            chart = SimaiParser.parse_simai_chart(request["text"], request["first"])
            parse_diagnostics = REPORT_WRITER.buf.getvalue()[parse_start:]
            failures = [
                line for line in parse_diagnostics.splitlines()
                if re.search(r"\b(?:invalid|unknown|error)\b", line, re.IGNORECASE)
            ]
            if failures:
                raise ValueError("Simai parse error: " + "\n".join(failures))
            actions = NoteActionConverter.generate_action(chart)
            total = len(chart)
            REPORT_WRITER.writeln("谱面加载完成，共%d个note" % total)
            REPORT_WRITER.writeln("========== 静态检查 ==========")
            static = StaticMuriChecker.check(chart)
            REPORT_WRITER.writeln("========== 动态检查 ==========")
            manager = JudgeManager()
            manager.load_chart(chart, actions)
            while manager.note_pointer < total or manager.active_notes:
                manager.tick(1)
            report = REPORT_WRITER.buf.getvalue()
        return {
            "static": static,
            "dynamic": manager.muri_record_list,
            "report": report,
            "metadata": {
                "upstream": "Minepig/MaiMuriDX",
                "upstream_commit": UPSTREAM_COMMIT,
                "first_seconds": request["first"],
                "note_count": total,
                "action_count": len(actions),
                "judge_tps": JUDGE_TPS,
                "simulation_step_ticks": 1,
                "static_delta_unit": "ticks (1/180 second)",
                "dynamic_time_unit": "seconds",
                "parse_status": "accepted by upstream parser and wrapper termination checks",
                "config": config,
            },
        }


if __name__ == "__main__":
    # Isolated mode ignores PYTHONIOENCODING; fix Windows pipe encoding here.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        payload = json.loads(sys.stdin.read())
        response = run(payload)
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False))
