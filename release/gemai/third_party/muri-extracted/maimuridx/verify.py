"""Meaningful fixture, isolation, and optional original-CLI parity checks."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
if __package__:
    from .api import DetectionError, UPSTREAM_COMMIT, detect_file, detect_simai, parse_maidata
else:
    from api import DetectionError, UPSTREAM_COMMIT, detect_file, detect_simai, parse_maidata

ROOT = Path(__file__).resolve().parent


def hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    }


def command(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result


def verify(original: Path | None = None) -> dict:
    before = hashes(ROOT)
    if original:
        commit = command(["git", "-C", str(original), "rev-parse", "HEAD"], ROOT).stdout.strip()
        assert commit == UPSTREAM_COMMIT
        for name in hashes(ROOT / "upstream"):
            assert (ROOT / "upstream" / name).read_bytes() == (original / name).read_bytes(), name
    cases = {}
    for fixture in sorted((ROOT / "fixtures").glob("*.simai")):
        result = detect_file(fixture)
        cases[fixture.stem] = result
    expected = {
        "clean": ([], []),
        "multitouch": ([], ["MultiTouch"]),
        "overlap": (["Overlap"], ["Overlap"]),
        "slide_fast": ([], ["SlideTooFast"]),
    }
    for case, (static, dynamic) in expected.items():
        assert [r["type"] for r in cases[case]["static"]] == static, case
        assert [r["type"] for r in cases[case]["dynamic"]] == dynamic, case
    assert "SlideHeadTap" in {r["type"] for r in cases["slide_head"]["static"]}
    assert "TapOnSlide" in {r["type"] for r in cases["tap_on_slide"]["static"]}
    assert all(r["metadata"]["config"]["hand_radius_normal"] == 60 for r in cases.values())
    for _ in range(3):
        assert detect_file(ROOT / "fixtures/overlap.simai") == cases["overlap"]
        assert detect_file(ROOT / "fixtures/clean.simai") == cases["clean"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        repeated = list(pool.map(detect_file, [ROOT / "fixtures/overlap.simai"] * 3))
    assert repeated == [cases["overlap"]] * 3
    selected = detect_file(ROOT / "fixtures/maidata.txt", difficulty=5)
    assert selected["metadata"]["first_seconds"] == 1.25
    assert selected["metadata"]["difficulty"] == "5"
    assert selected["dynamic"][0]["type"] == "MultiTouch"
    assert abs(selected["dynamic"][0]["time"] - 1.25) < 1e-10
    clean_selected = detect_file(ROOT / "fixtures/maidata.txt", difficulty=4)
    assert clean_selected["metadata"]["note_count"] == 4
    assert clean_selected["static"] == clean_selected["dynamic"] == []
    assert detect_file(ROOT / "fixtures/maidata.txt", difficulty=5, first=2.5)["dynamic"][0]["time"] == 2.5
    try:
        detect_file(ROOT / "fixtures/maidata.txt")
    except ValueError as exc:
        assert "4, 5" in str(exc)
    else:
        raise AssertionError("ambiguous maidata silently selected a difficulty")
    fields = parse_maidata("&first=2&inote_5=(120){4}1,&first_5=3")
    assert fields["first"] == "2" and fields["first_5"] == "3"
    assert detect_simai("(120){4}1,", hand_radius=75)["metadata"]["config"]["hand_radius_normal"] == 75
    assert detect_simai("(120){4}1,")["metadata"]["config"]["hand_radius_normal"] == 60
    try:
        detect_simai("(120){4}1,", config={"unknown": 1})
    except DetectionError as exc:
        assert "Unknown config key" in str(exc)
    else:
        raise AssertionError("unknown configuration accepted")
    invalid_inputs = [
        "",
        " \n",
        "|| only a comment\n",
        "(120){4}1",
        "(120){4}1/",
        "(120){4}1/2",
        "(120){4}1/,",
        "(120){4}1-5[4:],E",
        "(0){4}1,E",
        "(120){4}X,",
        "(120){4}1-2[4:1],E",
        "(120){4}1(150),",
    ]
    for invalid in invalid_inputs:
        try:
            detect_simai(invalid)
        except DetectionError:
            pass
        else:
            raise AssertionError(f"malformed chart was silently accepted: {invalid!r}")
    for valid in ("(120){4}1,E", "(120){4}1,E || end", "(120){4}1,|| comment\nE\n", "(120){4},,E"):
        detect_simai(valid)
    with tempfile.TemporaryDirectory(prefix="maimuridx-verification-") as directory:
        other_cwd = Path(directory)
        # Import the API by absolute file path while the current directory is empty.
        script = (
            "import importlib.util; "
            f"spec=importlib.util.spec_from_file_location('muri_api',{str(ROOT / 'api.py')!r}); "
            "api=importlib.util.module_from_spec(spec); spec.loader.exec_module(api); "
            "r=api.detect_simai('(120){4}1,'); "
            "assert not r['static'] and not r['dynamic']; print('ok')"
        )
        assert command([sys.executable, "-B", "-c", script], other_cwd).stdout.strip() == "ok"
        assert list(other_cwd.iterdir()) == [], "API wrote into caller cwd"
        comment_file = other_cwd / "comment-fields.simai"
        comment_file.write_text("(120){4}1,|| &inote_5=(120){4}123,\nE", encoding="utf-8")
        comment_result = detect_file(comment_file)
        assert comment_result["metadata"]["format"] == "simai"
        assert comment_result["metadata"]["note_count"] == 1
        assert comment_result["static"] == comment_result["dynamic"] == []
        json_path, report_path = other_cwd / "result.json", other_cwd / "report.txt"
        command([
            sys.executable, "-B", str(ROOT / "cli.py"),
            str(ROOT / "fixtures/overlap.simai"),
            "-j", str(json_path), "-o", str(report_path),
        ], other_cwd)
        cross_cwd = json.loads(json_path.read_text(encoding="utf-8"))
        assert cross_cwd == cases["overlap"]
        assert report_path.read_text(encoding="utf-8") == cases["overlap"]["report"]
        first_file = other_cwd / "maidata.txt"
        first_file.write_text("&first=1.25\n&first_5=2.5\n&inote_5=(120){4}1/3/5,\n", encoding="utf-8")
        assert detect_file(first_file)["metadata"]["first_seconds"] == 2.5
        invalid_file = other_cwd / "invalid.simai"
        invalid_file.write_text("(120){4}1-5[4:],E", encoding="utf-8")
        rejected_json = other_cwd / "invalid.json"
        rejected = subprocess.run(
            [sys.executable, "-B", str(ROOT / "cli.py"), str(invalid_file), "-j", str(rejected_json)],
            cwd=other_cwd, capture_output=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert rejected.returncode == 2
        assert "parse error" in rejected.stderr.lower()
        assert not rejected_json.exists(), "failed parse produced a successful JSON result"
        if original:
            for case, result in cases.items():
                original_json = other_cwd / f"{case}.upstream.json"
                original_report = other_cwd / f"{case}.upstream.txt"
                command([
                    sys.executable, "-B", str(original / "cli.py"),
                    "-f", str(ROOT / "fixtures" / f"{case}.simai"),
                    "-j", str(original_json), "-o", str(original_report),
                ], other_cwd)
                records = json.loads(original_json.read_text(encoding="utf-8"))
                assert records == {"static": result["static"], "dynamic": result["dynamic"]}, case
                assert original_report.read_text(encoding="utf-8") == result["report"], case
    for file in list(ROOT.glob("*.py")) + list((ROOT / "upstream").rglob("*.py")):
        ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file), feature_version=(3, 10))
    assert hashes(ROOT) == before, "detector source, fixtures, or config changed"
    return {
        "python": sys.version,
        "upstream_commit": UPSTREAM_COMMIT,
        "core_byte_identity": bool(original),
        "original_cli_parity": len(cases) if original else "not requested",
        "repeated_calls": 6,
        "parallel_calls": 3,
        "cwd_independence": True,
        "config_and_source_unchanged": True,
        "maidata_difficulty_and_first": True,
        "python_3_10_grammar": True,
        "invalid_inputs_rejected": len(invalid_inputs),
        "conventional_final_E_accepted": True,
        "parse_failure_cli_exit2_no_result": True,
        "maidata_fields_in_simai_comments_ignored": True,
        "cases": {
            case: {
                "static": [r["type"] for r in result["static"]],
                "dynamic": [r["type"] for r in result["dynamic"]],
                "note_count": result["metadata"]["note_count"],
            }
            for case, result in cases.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, help="original MaiMuriDX checkout for byte identity and CLI parity")
    args = parser.parse_args()
    print(json.dumps(verify(args.original.resolve() if args.original else None), ensure_ascii=False, indent=2))
